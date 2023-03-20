import requests
import argparse
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_argvs():
    parser = argparse.ArgumentParser(description='asr client')
    parser.add_argument("--processes", help="processes num", type=int, default=1)
    parser.add_argument("--threads", help="threads num", type=int, default=1)
    parser.add_argument("--times", help="test times", type=int, default=1)
    parser.add_argument("--server_address", help="server_address", type=str,
                        default="https://aigc.wair.ac.cn")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()
    logger.warning("[main] server_address: {}".format(args.server_address))

    request_data = {
        "inputs": [{
            "name": "input__0",
            "shape": [2, 3],
            "datatype": "INT64",
            "data": [[1, 2, 3], [4, 5, 6]]
        }],
        "outputs": [{"name": "output__0"}, {"name": "output__1"}]
    }

    response = requests.post(url="http://localhost:8000/v2/models/fc_model_pt/versions/1/infer", json=request_data)
    logger.info("response: {}, dict: {}".format(response, response.__dict__))

    if response.status_code == 200:
        logger.info("status_code: {}, response: {}".format(response.status_code, response))
        r_text = json.loads(response.text)
        logger.info("text: {}".format(response.text))
    else:
        logger.error("status_code: {}, response: {}".format(response.status_code, response))
