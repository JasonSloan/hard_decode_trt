
#include "cuvid_decoder.hpp"
#include "../utils/cuda_tools.hpp"
#include <nvcuvid.h>
#include <mutex>
#include <vector>
#include <sstream>
#include <string.h>
#include <assert.h>
#include <opencv2/opencv.hpp>

using namespace std;

namespace FFHDDecoder{
    static float GetChromaHeightFactor(cudaVideoSurfaceFormat eSurfaceFormat)
    {
        float factor = 0.5;
        switch (eSurfaceFormat)
        {
        case cudaVideoSurfaceFormat_NV12:
        case cudaVideoSurfaceFormat_P016:
            factor = 0.5;
            break;
        case cudaVideoSurfaceFormat_YUV444:
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            factor = 1.0;
            break;
        }

        return factor;
    }

    static int GetChromaPlaneCount(cudaVideoSurfaceFormat eSurfaceFormat)
    {
        int numPlane = 1;
        switch (eSurfaceFormat)
        {
        case cudaVideoSurfaceFormat_NV12:
        case cudaVideoSurfaceFormat_P016:
            numPlane = 1;
            break;
        case cudaVideoSurfaceFormat_YUV444:
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            numPlane = 2;
            break;
        }

        return numPlane;
    }

    IcudaVideoCodec ffmpeg2NvCodecId(int ffmpeg_codec_id) {
        switch (ffmpeg_codec_id) {
            /*AV_CODEC_ID_MPEG1VIDEO*/ case 1   : return cudaVideoCodec_MPEG1;        
            /*AV_CODEC_ID_MPEG2VIDEO*/ case 2   : return cudaVideoCodec_MPEG2;        
            /*AV_CODEC_ID_MPEG4*/ case 12       : return cudaVideoCodec_MPEG4;        
            /*AV_CODEC_ID_VC1*/ case 70         : return cudaVideoCodec_VC1;          
            /*AV_CODEC_ID_H264*/ case 27        : return cudaVideoCodec_H264;         
            /*AV_CODEC_ID_HEVC*/ case 173       : return cudaVideoCodec_HEVC;         
            /*AV_CODEC_ID_VP8*/ case 139        : return cudaVideoCodec_VP8;          
            /*AV_CODEC_ID_VP9*/ case 167        : return cudaVideoCodec_VP9;          
            /*AV_CODEC_ID_MJPEG*/ case 7        : return cudaVideoCodec_JPEG;         
            default                             : return cudaVideoCodec_NumCodecs;
        }
    }

    class CUVIDDecoderImpl : public CUVIDDecoder{
    public:
        bool create(bool bUseDeviceFrame, int gpu_id, cudaVideoCodec eCodec, bool bLowLatency = false,
                const CropRect *pCropRect = nullptr, const ResizeDim *pResizeDim = nullptr, int max_cache = -1,
                int maxWidth = 0, int maxHeight = 0, unsigned int clkRate = 1000)
            {
            // 是否使用显存存储解码后的视频帧
            m_bUseDeviceFrame = bUseDeviceFrame;
            // 设置视频编码类型
            m_eCodec = eCodec;
            // 设置最大视频宽度
            m_nMaxWidth = maxWidth;
            // 设置最大视频高度
            m_nMaxHeight = maxHeight;
            // 设置最大缓存帧数
            m_nMaxCache  = max_cache;
            // 设置使用的 GPU 设备 ID
            m_gpuID      = gpu_id;
            
            // 如果 m_gpuID 为 -1，表示使用当前设备，获取当前设备的 ID 并赋值给 m_gpuID
            if(m_gpuID == -1) checkCudaRuntime(cudaGetDevice(&m_gpuID));
            
            // 创建一个 AutoDevice 对象，用于自动管理 CUDA 设备上下文切换。AutoDevice 类应在 CUDATools 命名空间中定义
            CUDATools::AutoDevice auto_device_exchange(m_gpuID);  
            // 如果传入的裁剪矩形指针不为空，将其内容复制到成员变量 m_cropRect 中
            if (pCropRect) m_cropRect = *pCropRect;
            // 如果传入的调整尺寸结构体指针不为空，将其内容复制到成员变量 m_resizeDim 中
            if (pResizeDim) m_resizeDim = *pResizeDim;
            // 定义一个 CUDA 上下文指针，用于存储当前的 CUDA 上下文
            CUcontext cuContext = nullptr;
            // 获取当前的 CUDA 上下文，并将其存储到 cuContext 中
            checkCudaDriver(cuCtxGetCurrent(&cuContext));
            
            // 如果当前 CUDA 上下文为空，输出错误信息并返回 false 表示创建失败
            if(cuContext == nullptr){
                INFOE("Current Context is nullptr.");
                return false;
            }
            
            // 创建一个 CUDA 视频上下文锁，用于同步对 CUDA 视频上下文的访问，若创建失败则返回 false
            if(!checkCudaDriver(cuvidCtxLockCreate(&m_ctxLock, cuContext))) return false;
            // 创建一个 CUDA 流，用于异步操作，若创建失败则返回 false
            if(!checkCudaRuntime(cudaStreamCreate(&m_cuvidStream))) return false;
            
            // 定义一个 CUDA 视频解析器参数结构体，并初始化为 0
            CUVIDPARSERPARAMS videoParserParameters = {};
            // 设置视频解析器要处理的视频编码类型
            videoParserParameters.CodecType = eCodec;                           
            // 设置视频解析器支持的最大解码表面数量为 1。解码表面用于存储解码后的视频帧数据，值越大占用显存越多，但并行化高、效率高
            // 此时设置只是临时设置，最终设置由回调函数handleVideoSequence返回，回调函数返回的值会最终设置并修改该参数
            videoParserParameters.ulMaxNumDecodeSurfaces = 1;                   
            // 设置时钟频率
            videoParserParameters.ulClockRate = clkRate;                        
            // 根据 bLowLatency 参数设置最大显示延迟。如果 bLowLatency 为 true，最大显示延迟为 0，即低延迟模式；否则为 1
            videoParserParameters.ulMaxDisplayDelay = bLowLatency ? 0 : 1;      
            // 将当前对象的指针赋值给 pUserData，这样在回调函数中可以通过该指针访问当前对象的成员
            videoParserParameters.pUserData = this;                             
            // 设置视频序列信息回调函数，当解析器解析到视频序列信息时，会调用 handleVideoSequenceProc 函数
            videoParserParameters.pfnSequenceCallback = handleVideoSequenceProc;
            // 设置图片解码回调函数，当解析器需要解码图片时，会调用 handlePictureDecodeProc 函数
            videoParserParameters.pfnDecodePicture = handlePictureDecodeProc;   
            // 设置图片显示回调函数，当解析器需要显示图片时，会调用 handlePictureDisplayProc 函数
            videoParserParameters.pfnDisplayPicture = handlePictureDisplayProc; 
            // 创建一个 CUDA 视频解析器，若创建失败则返回 false
            if(!checkCudaDriver(cuvidCreateVideoParser(&m_hParser, &videoParserParameters))) return false;
            // 所有操作成功，返回 true 表示创建成功
            return true;
        }

        int decode(const uint8_t *pData, int nSize, int64_t nTimestamp=0) override
        {
            // 重置已解码的帧数为 0，用于统计本次解码过程中解码的帧数
            m_nDecodedFrame = 0;
            // 重置已返回的解码帧数为 0，用于记录已经返回给调用者的解码帧数
            m_nDecodedFrameReturned = 0;
            // 定义一个 CUDA 视频源数据包结构体，并初始化为 0
            CUVIDSOURCEDATAPACKET packet = { 0 };
            // 将传入的视频数据指针赋值给数据包的有效负载指针
            packet.payload = pData;
            // 将传入的视频数据大小赋值给数据包的有效负载大小
            packet.payload_size = nSize;
            // 设置数据包的标志位，表示数据包包含时间戳信息
            packet.flags = CUVID_PKT_TIMESTAMP;
            // 将传入的时间戳赋值给数据包的时间戳字段
            packet.timestamp = nTimestamp;
            // 检查传入的视频数据指针是否为空或者数据大小是否为 0
            if (!pData || nSize == 0) {
                // 如果数据为空或者数据大小为 0，设置数据包的标志位，表示视频流结束
                packet.flags |= CUVID_PKT_ENDOFSTREAM;
            }

            try{
                // 创建一个 AutoDevice 对象，用于自动管理 CUDA 设备上下文切换到指定的 GPU 设备
                CUDATools::AutoDevice auto_device_exchange(m_gpuID);
                // 调用 cuvidParseVideoData 函数解析视频数据包，如果解析失败则返回 -1
                if(!checkCudaDriver(cuvidParseVideoData(m_hParser, &packet)))
                    return -1;
            }catch(...){
                // 捕获所有异常，若捕获到异常则返回 -1，表示解码过程中出现错误
                return -1;
            }

            m_iFrameIndex++;
            // 解析成功，返回已解码的帧数
            return m_nDecodedFrame;
        }

        static int CUDAAPI handleVideoSequenceProc(void *pUserData, CUVIDEOFORMAT *pVideoFormat) { return ((CUVIDDecoderImpl *)pUserData)->handleVideoSequence(pVideoFormat); }
        static int CUDAAPI handlePictureDecodeProc(void *pUserData, CUVIDPICPARAMS *pPicParams) { return ((CUVIDDecoderImpl *)pUserData)->handlePictureDecode(pPicParams); }
        static int CUDAAPI handlePictureDisplayProc(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo) { return ((CUVIDDecoderImpl *)pUserData)->handlePictureDisplay(pDispInfo); }
        
        // 该回调函数只有解析器在初始序列头或遇到视频格式更改时触发此回调
        // 主要实现获得当前gpu解码能力，根据当前视频流信息，判断是否超过当前gpu解码能力，然后创建解码器
        int handleVideoSequence(CUVIDEOFORMAT *pVideoFormat){
            // 从视频格式信息中获取最小解码表面数量，并将其赋值给变量 nDecodeSurface
            // 解码表面用于存储解码后的视频帧数据，此值由视频格式决定
            int nDecodeSurface = pVideoFormat->min_num_decode_surfaces;
            // 定义一个 CUVIDDECODECAPS 结构体变量 decodecaps
            // 该结构体用于存储 CUDA 视频解码的能力信息，如支持的编解码器、分辨率等
            CUVIDDECODECAPS decodecaps;
            // 使用 memset 函数将 decodecaps 结构体的内存区域初始化为 0
            // 确保结构体中的所有成员都被正确初始化为已知状态，避免未定义行为
            memset(&decodecaps, 0, sizeof(decodecaps));

            // 将当前视频格式的编码类型赋值给 decodecaps 结构体的 eCodecType 成员
            // 用于指定要查询解码能力的视频编码类型
            decodecaps.eCodecType = pVideoFormat->codec;
            // 将当前视频格式的色度格式赋值给 decodecaps 结构体的 eChromaFormat 成员
            // 用于指定要查询解码能力的视频色度格式
            decodecaps.eChromaFormat = pVideoFormat->chroma_format;
            // 将当前视频格式的亮度位深度减去 8 后的值赋值给 decodecaps 结构体的 nBitDepthMinus8 成员
            // 用于指定要查询解码能力的视频亮度位深度信息
            decodecaps.nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;

            // 调用 cuvidGetDecoderCaps 函数获取当前 GPU 对于指定视频编码、色度格式和位深度的解码能力
            // 并将结果存储在 decodecaps 结构体中。如果调用失败，checkCudaDriver 会进行错误处理
            checkCudaDriver(cuvidGetDecoderCaps(&decodecaps));

            // 检查当前 GPU 是否支持指定的视频解码参数
            // 如果 bIsSupported 为 false，表示当前 GPU 不支持该视频的解码
            if(!decodecaps.bIsSupported){
                // 若不支持，抛出一个运行时异常，提示该编解码器在当前 GPU 上不受支持
                throw std::runtime_error("Codec not supported on this GPU");
                // 返回最小解码表面数量
                return nDecodeSurface;
            }

            // 检查当前视频的编码宽度或高度是否超过 GPU 支持的最大分辨率
            if ((pVideoFormat->coded_width > decodecaps.nMaxWidth) ||
                (pVideoFormat->coded_height > decodecaps.nMaxHeight)){

                // 创建一个字符串流对象，用于构建错误信息
                std::ostringstream errorString;
                // 向字符串流中添加当前视频的分辨率信息
                errorString << std::endl
                            << "Resolution          : " << pVideoFormat->coded_width << "x" << pVideoFormat->coded_height << std::endl
                            // 向字符串流中添加 GPU 支持的最大分辨率信息
                            << "Max Supported (wxh) : " << decodecaps.nMaxWidth << "x" << decodecaps.nMaxHeight << std::endl
                            // 向字符串流中添加分辨率不支持的错误提示
                            << "Resolution not supported on this GPU";

                // 将字符串流中的内容转换为标准字符串
                const std::string cErr = errorString.str();
                // 抛出运行时异常，包含构建好的错误信息
                throw std::runtime_error(cErr);
                // 返回最小解码表面数量
                return nDecodeSurface;
            }

            // 检查当前视频的宏块数量（将编码宽度和高度右移 4 位后相乘）是否超过 GPU 支持的最大宏块数量
            if ((pVideoFormat->coded_width>>4)*(pVideoFormat->coded_height>>4) > decodecaps.nMaxMBCount){

                // 创建一个字符串流对象，用于构建错误信息
                std::ostringstream errorString;
                // 向字符串流中添加当前视频的宏块数量信息
                errorString << std::endl
                            << "MBCount             : " << (pVideoFormat->coded_width >> 4)*(pVideoFormat->coded_height >> 4) << std::endl
                            // 向字符串流中添加 GPU 支持的最大宏块数量信息
                            << "Max Supported mbcnt : " << decodecaps.nMaxMBCount << std::endl
                            // 向字符串流中添加宏块数量不支持的错误提示
                            << "MBCount not supported on this GPU";

                // 将字符串流中的内容转换为标准字符串
                const std::string cErr = errorString.str();
                // 抛出运行时异常，包含构建好的错误信息
                throw std::runtime_error(cErr);
                // 返回最小解码表面数量
                return nDecodeSurface;
            }

            // 重新设置视频编码类型，可能是为了纠正之前设置的值
            m_eCodec = pVideoFormat->codec;
            // 设置视频的色度格式
            m_eChromaFormat = pVideoFormat->chroma_format;
            // 设置视频亮度位深度减去 8 后的值
            m_nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
            // 根据亮度位深度设置每个像素的字节数，大于 0 则为 2 字节，否则为 1 字节
            m_nBPP = m_nBitDepthMinus8 > 0 ? 2 : 1;

            // 根据色度格式设置输出表面格式
            if (m_eChromaFormat == cudaVideoChromaFormat_420)
                // 根据亮度位深度选择不同的输出表面格式
                m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
            else if (m_eChromaFormat == cudaVideoChromaFormat_444)
                // 根据亮度位深度选择不同的输出表面格式
                m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
            else if (m_eChromaFormat == cudaVideoChromaFormat_422)
                // 目前不支持 4:2:2 输出格式，默认设置为 420 格式
                m_eOutputFormat = cudaVideoSurfaceFormat_NV12;  

            // 检查所选的输出格式是否被 GPU 支持
            if (!(decodecaps.nOutputFormatMask & (1 << m_eOutputFormat)))
            {
                // 若不支持，尝试选择其他支持的输出格式
                if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
                    m_eOutputFormat = cudaVideoSurfaceFormat_NV12;
                else if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
                    m_eOutputFormat = cudaVideoSurfaceFormat_P016;
                else if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
                    m_eOutputFormat = cudaVideoSurfaceFormat_YUV444;
                else if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
                    m_eOutputFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
                else 
                    // 若没有支持的输出格式，抛出运行时异常
                    throw std::runtime_error("No supported output format found");
            }
            // 保存当前视频格式信息
            m_videoFormat = *pVideoFormat;

            // 初始化 CUVIDDECODECREATEINFO 结构体，用于创建 CUDA 视频解码器
            CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
            // 设置视频编码类型
            videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
            // 设置视频的色度格式
            videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
            // 设置输出表面格式
            videoDecodeCreateInfo.OutputFormat = m_eOutputFormat;
            // 设置视频亮度位深度减去 8 后的值
            videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
            // 根据视频是否为逐行扫描设置去隔行模式
            if (pVideoFormat->progressive_sequence)
                videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
            else
                videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
            // 设置输出表面数量为 2，实现双缓冲机制
            videoDecodeCreateInfo.ulNumOutputSurfaces = 2;                  
            // 设置创建标志，优先使用 CUVID 进行解码
            videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
            // 设置解码表面数量为最大解码表面数量
            videoDecodeCreateInfo.ulNumDecodeSurfaces = nDecodeSurface;  
            // 设置视频上下文锁
            videoDecodeCreateInfo.vidLock = m_ctxLock;
            // 设置视频编码宽度
            videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
            // 设置视频编码高度
            videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
            // 更新最大视频宽度
            if (m_nMaxWidth < (int)pVideoFormat->coded_width)
                m_nMaxWidth = pVideoFormat->coded_width;
            // 更新最大视频高度
            if (m_nMaxHeight < (int)pVideoFormat->coded_height)
                m_nMaxHeight = pVideoFormat->coded_height;
            // 设置最大视频宽度
            videoDecodeCreateInfo.ulMaxWidth = m_nMaxWidth;
            // 设置最大视频高度
            videoDecodeCreateInfo.ulMaxHeight = m_nMaxHeight;

            // 检查是否没有裁剪和调整尺寸的需求
            if (!(m_cropRect.r && m_cropRect.b) && !(m_resizeDim.w && m_resizeDim.h)) {
                // 设置视频显示宽度
                m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
                // 设置视频亮度部分的高度
                m_nLumaHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
                // 设置输出表面的目标宽度
                videoDecodeCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
                // 设置输出表面的目标高度
                videoDecodeCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
            } else {
                // 若有调整尺寸的需求
                if (m_resizeDim.w && m_resizeDim.h) {
                    // 复制原始视频的显示区域信息
                    videoDecodeCreateInfo.display_area.left = pVideoFormat->display_area.left;
                    videoDecodeCreateInfo.display_area.top = pVideoFormat->display_area.top;
                    videoDecodeCreateInfo.display_area.right = pVideoFormat->display_area.right;
                    videoDecodeCreateInfo.display_area.bottom = pVideoFormat->display_area.bottom;
                    // 设置调整后的视频宽度
                    m_nWidth = m_resizeDim.w;
                    // 设置调整后的视频亮度部分的高度
                    m_nLumaHeight = m_resizeDim.h;
                }

                // 若有裁剪需求
                if (m_cropRect.r && m_cropRect.b) {
                    // 设置裁剪后的显示区域信息
                    videoDecodeCreateInfo.display_area.left = m_cropRect.l;
                    videoDecodeCreateInfo.display_area.top = m_cropRect.t;
                    videoDecodeCreateInfo.display_area.right = m_cropRect.r;
                    videoDecodeCreateInfo.display_area.bottom = m_cropRect.b;
                    // 设置裁剪后的视频宽度
                    m_nWidth = m_cropRect.r - m_cropRect.l;
                    // 设置裁剪后的视频亮度部分的高度
                    m_nLumaHeight = m_cropRect.b - m_cropRect.t;
                }
                // 设置输出表面的目标宽度
                videoDecodeCreateInfo.ulTargetWidth = m_nWidth;
                // 设置输出表面的目标高度
                videoDecodeCreateInfo.ulTargetHeight = m_nLumaHeight;
            }

            // 根据输出格式计算色度部分的高度
            m_nChromaHeight = (int)(m_nLumaHeight * GetChromaHeightFactor(m_eOutputFormat));
            // 根据输出格式获取色度平面的数量
            m_nNumChromaPlanes = GetChromaPlaneCount(m_eOutputFormat);
            // 设置映射表面的高度
            m_nSurfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
            // 设置映射表面的宽度
            m_nSurfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
            // 设置显示区域的底部坐标
            m_displayRect.b = videoDecodeCreateInfo.display_area.bottom;
            // 设置显示区域的顶部坐标
            m_displayRect.t = videoDecodeCreateInfo.display_area.top;
            // 设置显示区域的左部坐标
            m_displayRect.l = videoDecodeCreateInfo.display_area.left;
            // 设置显示区域的右部坐标
            m_displayRect.r = videoDecodeCreateInfo.display_area.right;

            // 创建 CUDA 视频解码器
            checkCudaDriver(cuvidCreateDecoder(&m_hDecoder, &videoDecodeCreateInfo));
            return nDecodeSurface;
        }

        /* 触发了实际的解码操作。不过，解码后的图片数据不会直接返回，而是存储在 CUDA 视频解码器管理的内部显存中。
        后续需要通过 cuvidMapVideoFrame 函数将解码后的帧映射到可访问的显存地址，再进行处理。*/
        int handlePictureDecode(CUVIDPICPARAMS *pPicParams){
            if (!m_hDecoder)
            {
                throw std::runtime_error("Decoder not initialized.");
                return false;
            }
            //INFO("handlePictureDecode CurrPicIdx = %d, m_nDecodePicCnt = %d", pPicParams->CurrPicIdx, m_nDecodePicCnt);
            m_nPicNumInDecodeOrder[pPicParams->CurrPicIdx] = m_nDecodePicCnt++;
            checkCudaDriver(cuvidDecodePicture(m_hDecoder, pPicParams));
            return 1;
        }

        // 主要功能是将解码后的视频帧从内部显存复制到用户指定的缓冲区（设备或主机内存），同时检查解码状态
        int handlePictureDisplay(CUVIDPARSERDISPINFO *pDispInfo){
            // 初始化 CUVIDPROCPARAMS 结构体，用于存储视频处理参数
            CUVIDPROCPARAMS videoProcessingParameters = {};
            // 设置是否为逐行帧，从显示信息中获取该标志
            videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
            // 设置是否为第二场，根据显示信息中的重复第一场标志计算得出
            videoProcessingParameters.second_field = pDispInfo->repeat_first_field + 1;
            // 设置是否顶场优先，从显示信息中获取该标志
            videoProcessingParameters.top_field_first = pDispInfo->top_field_first;
            // 设置是否为未配对场，当重复第一场标志小于 0 时表示未配对场
            videoProcessingParameters.unpaired_field = pDispInfo->repeat_first_field < 0;
            // 设置输出流，使用类成员中的 CUDA 流进行异步操作
            videoProcessingParameters.output_stream = m_cuvidStream;

            // 定义一个 CUDA 设备指针，用于存储映射后的视频帧的设备地址
            CUdeviceptr dpSrcFrame = 0;
            // 定义一个无符号整数，用于存储映射后视频帧每行的字节数
            unsigned int nSrcPitch = 0;
            // pDispInfo->picture_index：当前要显示的视频帧在解码表面（Decode Surfaces）中的索引
            // 调用 cuvidMapVideoFrame 函数将指定索引的视频帧映射到设备内存，获取其地址和每行字节数
            checkCudaDriver(cuvidMapVideoFrame(m_hDecoder, pDispInfo->picture_index, &dpSrcFrame,
                &nSrcPitch, &videoProcessingParameters));

            // 定义一个 CUVIDGETDECODESTATUS 结构体，用于存储解码状态信息
            CUVIDGETDECODESTATUS DecodeStatus;
            // 使用 memset 函数将 DecodeStatus 结构体的内存区域初始化为 0
            memset(&DecodeStatus, 0, sizeof(DecodeStatus));

            // 调用 cuvidGetDecodeStatus 函数获取指定索引视频帧的解码状态
            CUresult result = cuvidGetDecodeStatus(m_hDecoder, pDispInfo->picture_index, &DecodeStatus);
            // 检查获取状态是否成功，并且解码状态是否为错误或错误隐藏状态
            if (result == CUDA_SUCCESS && (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error || DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed))
            {
                // 若解码出错，打印出错视频帧在解码顺序中的编号
                INFOE("Decode Error occurred for picture %d\n", m_nPicNumInDecodeOrder[pDispInfo->picture_index]);
            }

            // 定义一个指向无符号字符的指针，用于存储解码后视频帧的地址
            uint8_t *pDecodedFrame = nullptr;
            {
                // 已解码帧数加 1，并检查是否超过存储解码后视频帧指针的向量大小
                if ((unsigned)++m_nDecodedFrame > m_vpFrame.size())
                {
                    // 如果超过了缓存限制，则覆盖最后一个图 
                    // 定义一个布尔变量，用于标记是否需要分配新的内存
                    bool need_alloc = true;
                    // 检查最大缓存帧数是否有限制
                    if(m_nMaxCache != -1){
                        // 若向量大小已达到最大缓存帧数
                        if(m_vpFrame.size() >= m_nMaxCache){
                            // 已解码帧数减 1
                            --m_nDecodedFrame;
                            // 不需要分配新内存
                            need_alloc = false;
                        }
                    }

                    // 若需要分配新内存
                    if(need_alloc){
                        // 定义一个指向无符号字符的指针，用于临时存储新分配的内存地址
                        uint8_t *pFrame = nullptr;
                        // 根据是否使用设备帧选择不同的内存分配方式
                        if (m_bUseDeviceFrame)
                            // 在设备内存中分配存储一帧视频数据所需大小的内存
                            checkCudaDriver(cuMemAlloc((CUdeviceptr *)&pFrame, get_frame_size()));
                        else
                            // 在主机内存中分配存储一帧视频数据所需大小的内存
                            checkCudaRuntime(cudaMallocHost(&pFrame, get_frame_size()));
                            
                        // 将新分配的内存地址添加到存储解码后视频帧指针的向量中
                        m_vpFrame.push_back(pFrame);
                        // 在存储时间戳的向量中添加初始时间戳 0
                        m_vTimestamp.push_back(0);
                    }
                }
                // 获取当前解码帧在向量中的地址
                pDecodedFrame = m_vpFrame[m_nDecodedFrame - 1];
                // 更新当前解码帧的时间戳
                m_vTimestamp[m_nDecodedFrame - 1] = pDispInfo->timestamp;
            } 

            // 初始化 CUDA_MEMCPY2D 结构体，用于进行二维内存复制操作
            CUDA_MEMCPY2D m = { 0 };
            // 设置源内存类型为设备内存
            m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            // 设置源设备内存地址为映射后的视频帧地址
            m.srcDevice = dpSrcFrame;
            // 设置源内存每行的字节数
            m.srcPitch = nSrcPitch; 
            // 根据是否使用设备帧设置目标内存类型
            m.dstMemoryType = m_bUseDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
            // 设置目标设备内存地址，同时更新 dstHost 指针
            m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame);
            // 设置目标内存每行的字节数
            m.dstPitch = m_nWidth * m_nBPP;
            // 设置复制区域的宽度（以字节为单位）
            m.WidthInBytes = m_nWidth * m_nBPP;
            // 设置复制区域的高度（亮度部分高度）
            m.Height = m_nLumaHeight;
            // 异步执行二维内存复制操作，将亮度部分数据从设备内存复制到目标内存
            checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));

            // 更新源设备内存地址，指向色度部分数据的起始位置
            m.srcDevice = (CUdeviceptr)((uint8_t *)dpSrcFrame + m.srcPitch * m_nSurfaceHeight);
            // 更新目标设备内存地址，指向存储色度部分数据的起始位置
            m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * m_nLumaHeight);
            // 设置复制区域的高度（色度部分高度）
            m.Height = m_nChromaHeight;
            // 异步执行二维内存复制操作，将第一部分色度数据从设备内存复制到目标内存
            checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));

            // 检查色度平面数量是否为 2
            if (m_nNumChromaPlanes == 2)
            {
                // 更新源设备内存地址，指向第二部分色度数据的起始位置
                m.srcDevice = (CUdeviceptr)((uint8_t *)dpSrcFrame + m.srcPitch * m_nSurfaceHeight * 2);
                // 更新目标设备内存地址，指向存储第二部分色度数据的起始位置
                m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * m_nLumaHeight * 2);
                // 设置复制区域的高度（色度部分高度）
                m.Height = m_nChromaHeight;
                // 异步执行二维内存复制操作，将第二部分色度数据从设备内存复制到目标内存
                checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));
            }
            
            // 若使用主机内存存储解码后的视频帧
            if(!m_bUseDeviceFrame){
                // 同步 CUDA 流，确保内存复制操作完成
                checkCudaDriver(cuStreamSynchronize(m_cuvidStream));
            }
            // 解除之前映射的视频帧，释放相关资源
            checkCudaDriver(cuvidUnmapVideoFrame(m_hDecoder, dpSrcFrame));
            // 函数返回 1 表示处理成功
            return 1;
        }

        virtual ICUStream get_stream() override{
            return m_cuvidStream;
        }

        int get_frame_size() override { assert(m_nWidth); return m_nWidth * (m_nLumaHeight + m_nChromaHeight * m_nNumChromaPlanes) * m_nBPP; }

        int get_width() override { assert(m_nWidth); return m_nWidth; }

        int get_height() override { assert(m_nLumaHeight); return m_nLumaHeight; }

        unsigned int get_frame_index() override { return m_iFrameIndex; }

        unsigned int get_num_decoded_frame() override {return m_nDecodedFrame;}

        cudaVideoSurfaceFormat get_output_format() { return m_eOutputFormat; }

        uint8_t* get_frame(int64_t* pTimestamp = nullptr, unsigned int* pFrameIndex = nullptr) override{
            if (m_nDecodedFrame > 0){
                if (pFrameIndex)
                    *pFrameIndex = m_iFrameIndex;

                if (pTimestamp)
                    *pTimestamp = m_vTimestamp[m_nDecodedFrameReturned];

                m_nDecodedFrame--;
                return m_vpFrame[m_nDecodedFrameReturned++];
            }
            return nullptr;
        }

        virtual ~CUVIDDecoderImpl(){
            
            if (m_hParser) 
                cuvidDestroyVideoParser(m_hParser);

            if (m_hDecoder) 
                cuvidDestroyDecoder(m_hDecoder);

            for (uint8_t *pFrame : m_vpFrame){
                if (m_bUseDeviceFrame)
                    cuMemFree((CUdeviceptr)pFrame);
                else
                    cudaFreeHost(pFrame);
            }
            cuvidCtxLockDestroy(m_ctxLock);
        }

    private:
        // CUDA 视频上下文锁，用于同步对 CUDA 视频上下文的访问，确保多线程环境下操作的线程安全
        CUvideoctxlock m_ctxLock = nullptr;     
        // CUDA 视频解析器句柄，用于解析输入的视频数据，将其拆分为可解码的单元    
        CUvideoparser m_hParser = nullptr;   
        // CUDA 视频解码器句柄，负责对解析后的视频数据进行解码操作       
        CUvideodecoder m_hDecoder = nullptr;
        // 标志位，指示是否使用设备端帧。true 表示使用设备内存存储解码后的帧，false 表示使用主机内存        
        bool m_bUseDeviceFrame = false;             
        // dimension of the output
        unsigned int m_nWidth = 0, m_nLumaHeight = 0, m_nChromaHeight = 0;
        // 色度平面的数量
        unsigned int m_nNumChromaPlanes = 0;        
        // height of the mapped surface 
        // 映射表面的高度
        int m_nSurfaceHeight = 0; 
        // 映射表面的宽度                  
        int m_nSurfaceWidth = 0;    
        // 视频编码类型，初始化为无效值                
        cudaVideoCodec m_eCodec = cudaVideoCodec_NumCodecs; 
        // 视频的色度格式
        cudaVideoChromaFormat m_eChromaFormat;   
        // 输出视频帧的表面格式   
        cudaVideoSurfaceFormat m_eOutputFormat;
        // 亮度位深度减去 8 后的值     
        int m_nBitDepthMinus8 = 0; 
        // 每个像素的字节数                 
        int m_nBPP = 1;     
        // 存储视频格式信息的结构体                        
        CUVIDEOFORMAT m_videoFormat = {}; 
        // 显示区域的裁剪矩形          
        CropRect m_displayRect = {};  
        // 互斥锁，用于线程同步              
        mutex m_lock;                               
        // stock of frames
        // 存储解码后视频帧指针的向量
        std::vector<uint8_t *> m_vpFrame;           
        // timestamps of decoded frames
        // 存储解码后视频帧时间戳的向量
        std::vector<int64_t> m_vTimestamp;       
        // 已解码的帧数和已返回的解码帧数   
        int m_nDecodedFrame = 0, m_nDecodedFrameReturned = 0;  
        // 解码图片的计数和按解码顺序排列的图片编号数组 
        int m_nDecodePicCnt = 0, m_nPicNumInDecodeOrder[32];
        // CUDA 流，用于异步操作
        CUstream m_cuvidStream = 0;
        // 裁剪矩形，用于裁剪视频帧
        CropRect m_cropRect = {};
        // 调整尺寸的结构体，用于调整视频帧的尺寸
        ResizeDim m_resizeDim = {};
        // 当前帧的索引
        unsigned int m_iFrameIndex = 0;
        // 最大缓存帧数，-1 表示无限制
        int m_nMaxCache = -1;
        // 使用的 GPU 设备 ID，-1 表示当前设备
        int m_gpuID = -1;
        // 最大视频宽度和高度
        unsigned int m_nMaxWidth = 0, m_nMaxHeight = 0;
    };

    std::shared_ptr<CUVIDDecoder> create_cuvid_decoder(
        bool bUseDeviceFrame,   // true: use device frame, false: use host frame
        IcudaVideoCodec eCodec, // codec type
        int max_cache,          // max number of frames to cache, -1 means no limit
        int gpu_id,             // gpu id, -1 means current device
        const CropRect *pCropRect, // crop rectangle, nullptr means no crop
        const ResizeDim *pResizeDim // resize dimensions, nullptr means no resize
    ){
        shared_ptr<CUVIDDecoderImpl> instance(new CUVIDDecoderImpl());
        if(!instance->create(bUseDeviceFrame, gpu_id, (cudaVideoCodec)eCodec, false, pCropRect, pResizeDim, max_cache))
            instance.reset();
        return instance;
    }
}; //FFHDDecoder
