import requests
import argparse
import logging

logger = logging.getLogger()


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

    res = requests.post(url="http://localhost:8000/v2/models/fc_model_pt/versions/1/infer", json=request_data).json()
    print(res)
