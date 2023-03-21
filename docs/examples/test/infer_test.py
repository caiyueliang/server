import requests
import argparse
import logging
import json
import numpy as np
from PIL import Image

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_densenet_onnx_data(image_path, new_size=(224, 224)):
    im = Image.open(fp=image_path)
    logger.warning("[get_densenet_onnx_data] im.size: {}".format(im.size))
    out = im.resize(new_size, Image.ANTIALIAS)
    logger.warning("[get_densenet_onnx_data] out.size: {}".format(out.size))

    out_arr = np.array(out)
    logger.warning("[get_densenet_onnx_data] out_arr.shape: {}".format(out_arr.shape))

    request_data = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 16],
                "datatype": "FP32",
                "data": [out_arr]
            }
        ],
        "outputs": [{"name": "OUTPUT0"}, {"name": "OUTPUT1"}]
    }

    return request_data


def get_inception_graphdef_data(image_path, new_size=(299, 299)):
    im = Image.open(fp=image_path)
    logger.warning("[get_inception_graphdef_data] im.size: {}".format(im.size))
    out = im.resize(new_size, Image.ANTIALIAS)
    logger.warning("[get_inception_graphdef_data] out.size: {}".format(out.size))

    out_arr = np.array(out)
    logger.warning("[get_inception_graphdef_data] out_arr.shape: {}".format(out_arr.shape))

    request_data = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 299, 299, 3],
                "datatype": "FP32",
                "data": [out_arr]
            }
        ],
        "outputs": [{"name": "InceptionV3/Predictions/Softmax"}]
    }

    return request_data


def parse_argvs():
    parser = argparse.ArgumentParser(description='asr client')
    parser.add_argument("--processes", help="processes num", type=int, default=1)
    parser.add_argument("--threads", help="threads num", type=int, default=1)
    parser.add_argument("--times", help="test times", type=int, default=1)
    # parser.add_argument("--url", type=str, default="http://localhost:8000/v2/models/fc_model_pt/versions/1/infer")
    parser.add_argument("--host", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="simple")
    parser.add_argument("--version", type=str, default="1")
    parser.add_argument("--image_path", type=str, default="./data/images/1.jpeg")

    args = parser.parse_args()
    logger.warning('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()
    # logger.warning("[main] server_address: {}".format(args.server_address))
    # http://localhost:8000/v2/models/simple/versions/1/infer
    base_url = "{}/v2/models/{}/versions/{}/infer"
    url = base_url.format(args.host, args.model, args.version)
    logger.warning("[main] url: {}".format(url))

    if args.model == "simple":
        request_data = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "shape": [1, 16],
                    "datatype": "INT32",
                    "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                },
                {
                    "name": "INPUT1",
                    "shape": [1, 16],
                    "datatype": "INT32",
                    "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                }
            ],
            "outputs": [{"name": "OUTPUT0"}, {"name": "OUTPUT1"}]
        }
    elif args.model == "densenet_onnx":
        request_data = get_densenet_onnx_data(image_path=args.image_path)
    elif args.model == "inception_graphdef":
        request_data = get_inception_graphdef_data(image_path=args.image_path)
    elif args.model == "simple_identity":
        request_data = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "shape": [1, 2],
                    "datatype": "BYTES",
                    "data": ["Hello", "WORLD"]
                }
            ],
            "outputs": [{"name": "OUTPUT0"}]
        }
    else:
        request_data = {
            "inputs": [{
                "name": "input__0",
                "shape": [2, 3],
                "datatype": "INT64",
                "data": [[1, 2, 3], [4, 5, 6]]
            }],
            "outputs": [{"name": "output__0"}, {"name": "output__1"}]
        }

    response = requests.post(url=url, json=request_data)
    logger.warning("response: {}, dict: {}".format(response, response.__dict__))

    if response.status_code == 200:
        logger.warning("status_code: {}, response: {}".format(response.status_code, response))
        r_text = json.loads(response.text)
        logger.warning("text: {}".format(response.text))
    else:
        logger.error("status_code: {}, response: {}".format(response.status_code, response))
