
#include <opencv2/opencv.hpp>
#include <utils/ilogger.hpp>
#include <ffhdd/ffmpeg_demuxer.hpp>
#include <ffhdd/cuvid_decoder.hpp>
#include <ffhdd/nalu.hpp>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

using namespace std;

struct DecodeInfo {
    int total_frames = 0;
    chrono::duration<double> duration;
};
mutex mtx;

static void test_hard_decode(string uri, vector<DecodeInfo>& decode_infos, int index) {
    auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri);
    if (demuxer == nullptr) {
        INFOE("demuxer create failed");
        return;
    }

    auto decoder = FFHDDecoder::create_cuvid_decoder(
        true, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, 0
    );

    if (decoder == nullptr) {
        INFOE("decoder create failed");
        return;
    }

    // 用来存储从解复用器（demuxer）获取的视频数据包的内存地址
    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    int64_t pts = 0;

    demuxer->get_extra_data(&packet_data, &packet_size);
    decoder->decode(packet_data, packet_size);

    string output_dir = "imgs_" + to_string(index);
    iLogger::rmtree(output_dir);
    iLogger::mkdir(output_dir);

    INFO("Start decode");
    /* 视频前几十帧可能解码失败(由于关键帧缺失，属于正常现象),
    但是如果视频有问题也可能会一直解码失败.
    所以这里设置一个变量ever_success, 只要成功解码过一帧就设置为true.
    所以只是前面几十帧解码失败，那么不会打印日志
    如果一直解码失败，那么会打印日志
    或者解码成功后，又解码失败会打印日志 */
    bool ever_success = false;
    auto start_time = std::chrono::high_resolution_clock::now();

    do {
        bool ret = demuxer->demux(&packet_data, &packet_size, &pts);
        int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
        if (!ret)                               // 解复用失败，打印日志
            INFOW("demuxer demux failed");
        if (ndecoded_frame > 0)                 
            ever_success = true;                // 只要成功解码过一帧就设置为true

        {
            lock_guard<mutex> lock(mtx);
            decode_infos[index].total_frames += ndecoded_frame;
        }

        for(int i = 0; i < ndecoded_frame; ++i){
            // /* 因为decoder获取的frame内存，是YUV-NV12格式的。储存内存大小是 [height * 1.5] * width byte
            //  因此构造一个height * 1.5,  width 大小的空间
            //  然后由opencv函数，把YUV-NV12转换到BGR，转换后的image则是正常的height, width, CV_8UC3 */
            // unsigned int frame_index = 0;
            // cv::Mat image(decoder->get_height() * 1.5, decoder->get_width(), CV_8U, decoder->get_frame(&pts, &frame_index));
            // cv::cvtColor(image, image, cv::COLOR_YUV2BGR_NV12);

            // // 直接使用解码器返回的 frame_index，避免索引不一致
            // INFO("write %s/img_%05d.jpg  %dx%d", output_dir.c_str(), frame_index, image.cols, image.rows);
            // cv::imwrite(cv::format("%s/img_%05d.jpg", output_dir.c_str(), frame_index), image);

            if (decoder->get_frame_index() > 1000 && ever_success == false)
                INFOE("Always failed");         // 如果一直解码失败，那么会打印日志
            if (decoder->get_frame_index() % 100 == 0)
                INFO("frame_index = %d", decoder->get_frame_index());
        }
    } while (packet_size > 0);

    auto end_time = std::chrono::high_resolution_clock::now();
    {
        lock_guard<mutex> lock(mtx);
        decode_infos[index].duration = end_time - start_time;
    }
}

int app_hard_decode() {
    // 并发测试多路视频(5060Ti 16G解码1920*1080可解80路, 帧率可保持在28fps以上)
    int n_videos = 1;
    vector<thread> threads;
    vector<DecodeInfo> decode_infos(n_videos);

    for (int i = 0; i < n_videos; ++i) 
        threads.emplace_back(test_hard_decode, "exp/0.mov", ref(decode_infos), i);

    for (auto& th : threads)
        if (th.joinable()) 
            th.join();

    // 在主线程中计算每个线程的平均 FPS 并打印
    for (int i = 0; i < n_videos; ++i) {
        double duration_seconds = decode_infos[i].duration.count();
        double avg_fps = decode_infos[i].total_frames / duration_seconds;
        INFO("Average FPS for exp/%d.mov: %.2f", i + 1, avg_fps);
    }

    return 0;
}