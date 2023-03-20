import requests
import argparse
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def parse_argvs():
    parser = argparse.ArgumentParser(description='asr client')
    parser.add_argument("--processes", help="processes num", type=int, default=1)
    parser.add_argument("--threads", help="threads num", type=int, default=1)
    parser.add_argument("--times", help="test times", type=int, default=1)
    # parser.add_argument("--url", type=str, default="http://localhost:8000/v2/models/fc_model_pt/versions/1/infer")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v2/models/simple/versions/1/infer")

    args = parser.parse_args()
    logger.warning('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()
    logger.warning("[main] server_address: {}".format(args.server_address))

    # request_data = {
    #     "inputs": [{
    #         "name": "input__0",
    #         "shape": [2, 3],
    #         "datatype": "INT64",
    #         "data": [[1, 2, 3], [4, 5, 6]]
    #     }],
    #     "outputs": [{"name": "output__0"}, {"name": "output__1"}]
    # }
    request_data = {
        "inputs": [
            {
                "name": "input0",
                "shape": [16],
                "datatype": "INT32",
                "data": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
            },
            {
                "name": "input1",
                "shape": [16],
                "datatype": "INT32",
                "data": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
            }
        ],
        "outputs": [{"name": "output0"}, {"name": "output1"}]
    }

    response = requests.post(url=args.url, json=request_data)
    logger.warning("response: {}, dict: {}".format(response, response.__dict__))

    if response.status_code == 200:
        logger.warning("status_code: {}, response: {}".format(response.status_code, response))
        r_text = json.loads(response.text)
        logger.warning("text: {}".format(response.text))
    else:
        logger.error("status_code: {}, response: {}".format(response.status_code, response))
