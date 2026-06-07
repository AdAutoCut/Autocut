import requests
import os



def download_csv_data(data_path: str) -> None:
    if os.path.exists(data_path):
        print(f"已下载，为 {data_path}")
        return
    
    # 可选环境（国内 / 新加坡）
    ENV = "prod"  # "prod" or "prod-sgp"

    # 域名映射
    ENV_URLS = {
        "prod": "http://kwaibi-sql-web.internal",
        "prod-sgp": "http://kwaibi-sql-sgp.internal"
    }

    # 请求地址
    url = f"{ENV_URLS[ENV]}/sql-access/api/v2/internal/task/download"

    # 请求参数
    params = {
        "principal": "", 
        "secretKey": "",   
        "id": ,   
        "fileType": "csv" 
    }

    # 发起 POST 请求
    response = requests.post(url, params=params)

    # 检查响应状态
    if response.status_code == 200:
        # 保存下载文件
        with open(data_path, "wb") as f:
            f.write(response.content)
        print(f"下载成功，已保存为 {data_path}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"响应内容: {response.text}")

download_csv_data("incycle_720_820.csv")