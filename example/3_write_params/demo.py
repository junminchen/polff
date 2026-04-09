import os
import time
from typing import Dict
from urllib.parse import urlparse
import requests
from tqdm import tqdm
BYTEFF_POL_SERVICE_URL = 'http://ep-13frkz496x3pc3n6nu4chqa1k.epsvc-36msh8kgxrvuo3w9p8g43gbo1.cn-beijing.privatelink.volces.com:8080'
BYTEFF_POL_EXTEND_SERVICE_URL = 'http://ep-13frkz496x3pc3n6nu4chqa1k.epsvc-36msh8kgxrvuo3w9p8g43gbo1.cn-beijing.privatelink.volces.com:8081'
def download_file(url: str, output_dir: str, retry_times: int = 3):
    """
    从给定的 URL 下载文件, 功能类似 curl -O, 并将文件保存到指定目录。
    """
    print(url)
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    for i in range(retry_times):
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 8192  # 8 KB chunks
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {filename}")
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                progress_bar.close()
                if total_size != 0 and progress_bar.n != total_size:
                    raise RuntimeError("错误, 下载的文件大小与预期不符!")
                return filepath
        except Exception as e:
            print(f"下载失败, 第 {i+1} 次重试")
            # 有可能服务器还没有挂载好文件, 等待 1 秒后重试
            time.sleep(1)
            if i == retry_times - 1:
                if os.path.exists(filepath):
                    os.remove(filepath)
                print(f"自动下载失败, 可以尝试手动 curl -O {url}")
                raise e
def send_with_smiles(smiles: Dict[str, str], model_type: str = 'origin', output_dir: str = os.getcwd()):
    """
    smiles (Dict[str, str]): 一个字典，键为 SMILES 字符串的名称，值为 SMILES 字符串。
        example:
            {
                "FSI": "[N-:1]([S:2](=[O:3])(=[O:4])[F:5])[S:6](=[O:7])(=[O:8])[F:9]",
            }
    model_type (str): 模型类型, 可选 'origin' 或 'extend', 默认 'origin'
    output_dir (str): 输出目录, 用于保存结果文件, 默认当前目录
    """
    assert len(smiles) == 1, "一次只能处理 1 个 SMILES 字符串"
    assert model_type in ['origin', 'extend'], "model_type 必须为 'origin' 或 'extend'"
    if model_type == 'origin':
        API_URL = f"{BYTEFF_POL_SERVICE_URL}/label/from-smiles"
    else:
        API_URL = f"{BYTEFF_POL_EXTEND_SERVICE_URL}/label/from-smiles"
    payload = {"smiles": smiles}
    try:
        print(f"向 {API_URL} 发送 POST 请求...")
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # 如果状态码不是 2xx，会抛出异常
        data = response.json()
        if len(data["processed_names"]) == len(smiles):
            print(f"计算全部成功, 开始下载结果到 {output_dir}")
            filepath = download_file(data["download_url"], output_dir)
            print(f"下载完成: {filepath}")
        elif len(data["processed_names"]) == 0:
            print("所有 SMILES 字符串均计算失败, 请检查是否有误")
        else:
            print(f"共有 {len(data['failed_names'])} 个 SMILES 字符串处理失败: {data['failed_names']}")
            print(f"开始下载处理成功的计算结果到 {output_dir}")
            filepath = download_file(data["download_url"], output_dir)
            print(f"下载完成: {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"请求失败:")
        print(f"{e}")
        if e.response is not None:
            print(f"状态码: {e.response.status_code}")
            print(f"服务器错误信息: {e.response.text}")
def query_quota(model_type: str):
    """
    查询当前已完成的任务里, 成功和失败的分子数量
    """
    assert model_type in ['origin', 'extend'], "model_type 必须为 'origin' 或 'extend'"
    if model_type == 'origin':
        API_URL = f"{BYTEFF_POL_SERVICE_URL}/query_quota"
    else:
        API_URL = f"{BYTEFF_POL_EXTEND_SERVICE_URL}/query_quota"
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()
    print(f"{model_type} 当前已完成的任务里, 成功: {data['succeed']}, 失败: {data['failed']}")
    return data

if __name__ == '__main__':
    if False:
        send_with_smiles(
            {
                "FSI": "[N-:1]([S:2](=[O:3])(=[O:4])[F:5])[S:6](=[O:7])(=[O:8])[F:9]",
            },
            model_type='origin',
            output_dir='./origin')

    if True:
        send_with_smiles({
            "SIO4": "CCCO[Si](C)(C)C",
        }, model_type='extend', output_dir='./extend')

    query_quota(model_type='origin')
    query_quota(model_type='extend')


    # ========== 20个扩展体系分子循环提交 ==========
    if True:
        from tqdm import tqdm

        # 20个扩展体系分子（非C,H,O,N,P,S,F,Cl,Li体系）
        # 保留必含分子 (Na, BF4, BOB, DFOB) + 新增TMOS和TTB
        extend_molecules = {
            # === 必含分子 ===
            'NA':        '[Na+:1]',
            'BF4':       '[B-:1]([F:2])([F:3])([F:4])([F:5])',
            'BOB':       '[B-:1]12([O:2][C:3](=[O:4])[C:5](=[O:6])[O:7]1)[O:8][C:9](=[O:10])[C:11](=[O:12])[O:13]2',
            'DFOB':      '[F:1][B-:2]1([F:9])[O:3][C:4](=[O:8])[C:5](=[O:7])[O:6]1',

            'TMOS': '[CH3:1][O:2][Si:3]([O:4][CH3:5])([O:6][CH3:7])[O:8][CH3:9]',
            'TMSB': '[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][B:6]([O:7][Si:8]([CH3:9])([CH3:10])[CH3:11])[O:12][Si:13]([CH3:14])([CH3:15])[CH3:16]',
            'TMB': '[CH3:1][O:2][B:3]([O:4][CH3:5])[O:6][CH3:7]',
            'TEB': '[CH3:1][CH2:2][O:3][B:4]([O:5][CH2:6][CH3:7])[O:8][CH2:9][CH3:10]',
            'HMDS': '[CH3:1][Si:2]([CH3:3])([CH3:4])[NH:5][Si:6]([CH3:7])([CH3:8])[CH3:9]',
            'TEOS': '[CH3:1][CH2:2][O:3][Si:4]([O:5][CH2:6][CH3:7])([O:8][CH2:9][CH3:10])[O:11][CH2:12][CH3:13]',
            'TMSPa': '[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][P:6](=[O:7])([O:8][Si:9]([CH3:10])([CH3:11])[CH3:12])[O:13][Si:14]([CH3:15])([CH3:16])[CH3:17]',
            'TMSPi': '[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][P:6]([O:7][Si:8]([CH3:9])([CH3:10])[CH3:11])[O:12][Si:13]([CH3:14])([CH3:15])[CH3:16]',
            'HMDSO': '[CH3:1][Si:2]([CH3:3])([CH3:4])[O:5][Si:6]([CH3:7])([CH3:8])[CH3:9]',
            'MTMS': '[CH3:1][O:2][Si:3]([CH3:4])([O:5][CH3:6])[O:7][CH3:8]',
            'TMOBX': '[CH3:1][O:2][B:3]1[O:4][B:5]([O:6][CH3:7])[O:8][B:9]([O:10][CH3:11])[O:12]1',
            'TPFPB': '[F:1][c:2]1[c:3]([F:4])[c:5]([F:6])[c:7]([B:8]([c:9]2[c:10]([F:11])[c:12]([F:13])[c:14]([F:15])[c:16]([F:17])[c:18]2[F:19])[c:20]2[c:21]([F:22])[c:23]([F:24])[c:25]([F:26])[c:27]([F:28])[c:29]2[F:30])[c:31]([F:32])[c:33]1[F:34]',
            'TMSNCS': '[CH3:1][Si:2]([CH3:3])([CH3:4])[N:5]=[C:6]=[S:7]',
        }
        # 'K':         '[K+:1]',
        # 'MG':        '[Mg2+:1]',
        # 'CA':        '[Ca2+:1]',
        # 'ZN':        '[Zn2+:1]',
        print("\n" + "="*60)
        print("开始循环提交20个扩展体系分子")
        print("="*60)

        output_dir = './extend_new'

        for mol_name, smiles in tqdm(extend_molecules.items(), desc="提交进度"):
            print(f"\n{'='*50}")
            print(f"正在提交: {mol_name}")
            print(f"SMILES: {smiles}")
            print(f"{'='*50}")

            try:
                send_with_smiles({mol_name: smiles}, model_type='extend', output_dir=output_dir)
            except Exception as e:
                print(f"跳过 {mol_name}, 错误: {e}")

            time.sleep(2)

        print("\n全部提交完成!")
        query_quota(model_type='extend')