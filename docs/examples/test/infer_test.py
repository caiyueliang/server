import requests


if __name__ == "__main__":
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
