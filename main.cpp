#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"
#include "openvino/core/preprocess/input_info.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/openvino.hpp"

void printInputAndOutputsInfo(const ov::Model& network) {
    std::cout << "model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        std::cout << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        std::cout << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        output shape: " << shape << std::endl;
    }
}

std::string readFileToString(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file) {
        throw std::runtime_error("파일을 열 수 없습니다: " + file_path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();  // 파일의 전체 내용을 스트림으로 읽어오기
    return buffer.str();     // std::string으로 반환
}

std::vector<uint8_t> readFileToBinary(const std::string& file_path) {
    // 바이너리 모드로 파일을 엽니다.
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("파일을 열 수 없습니다: " + file_path);
    }

    // 파일 크기를 구하고, 해당 크기만큼의 버퍼를 할당합니다.
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("파일을 읽는 데 실패했습니다: " + file_path);
    }

    return buffer;
}

int main(int argc, char const* argv[]) {
    std::cout << "Hello World" << std::endl;
    cv::Mat dog = cv::imread("C:\\Users\\brightro99\\Desktop\\Workspace\\OpenVINO_2024\\images\\dog.jpg");

    std::string weights_path = "C:\\Users\\brightro99\\Desktop\\Workspace\\OpenVINO_2024\\models\\pe-01.bin";
    std::string xml_path = "C:\\Users\\brightro99\\Desktop\\Workspace\\OpenVINO_2024\\models\\pe-01.xml";

    ov::Core core;

    try {
        std::string xml = readFileToString(xml_path);
        std::vector<uint8_t> weights = readFileToBinary(weights_path);

        ov::Tensor weights_tensor = ov::Tensor(ov::element::u8, {1, weights.size()}, weights.data());
        std::shared_ptr<ov::Model> model = core.read_model(xml, weights_tensor);
        printInputAndOutputsInfo(*model);

        ov::preprocess::PrePostProcessor ppp(model);

        ppp.input().tensor().set_layout("NCHW");
        ppp.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess().mean({0, 0, 0});
        ppp.input().preprocess().scale({255, 255, 255});
        ppp.input().preprocess().

    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
