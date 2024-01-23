import time
import requests
import pickle
import os
import pandas as pd
import random


url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/concepts/name/JSON?name=OLAPARIB"


# http 2.0 get request
def read_files(path):
    data = None
    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            # print(file)
            try:
                temp = pd.read_csv(path + file, sep="\t")
            except:  # noqa: E722
                temp = pd.read_csv(path + file, sep="\t", encoding="ISO-8859-1")

            if data is not None:
                data = pd.concat([data, temp], axis=0)
            else:
                data = temp

    data = data.reset_index(drop=True)
    return data


perturbed_data_treat_summ = read_files(
    "./datasets/row/cppa/CPPA_v1.0_Cell_line_perturbed_responses_p0_p1_TreatmentSummary/"
)  # noqa: E501

drug_names = perturbed_data_treat_summ["compound_name_1"].unique().tolist()
temp = []
for drug_name in drug_names:
    if not isinstance(drug_name, str):
        temp.append(drug_name)


for key in temp:
    drug_names.remove(key)

# drug_names 排序
drug_names.sort()

drug_smiles = {}
for drug_name in drug_names:
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/concepts/name/JSON?name=" + str(
        drug_name
    )
    response = requests.get(url)

    json_data_cid = response.json()
    try:
        CID = json_data_cid["ConceptsAndCIDs"]["CID"]
    except:  # noqa: E722
        if "PUGREST.NotFound" in response.text or "ConceptName" in response.text:
            print(drug_name + " 没有找到数据")
            continue
        else:
            print(json_data_cid)
            print(drug_name + " 未知错误")
            exit()

    if len(CID) >= 1:
        header = {
            "Cookie": "ncbi_sid=90C5FA7943107D01_0000SID",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0",  # noqa: E501
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://pubchem.ncbi.nlm.nih.gov/",
            "Content-Type": "application/json; charset=utf-8",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Te": "trailers",
        }
        url = (
            'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=json&query={"select":"*","collection":"compound","where":{"ands":[{"cid":"'
            + str(CID[0])
            + '"}]},"order":["cid,asc"],"start":1,"limit":10,"width":1000000,"listids":0}'
        )  # noqa: E501
        proxy = {"http": "http://127.0.0.1:8080"}
        response = requests.get(url, headers=header)
        try:
            json_data_smiles = response.json()

            for info in json_data_smiles["SDQOutputSet"][0]["rows"]:
                if info["cid"] == CID[0]:
                    print(drug_name, " canonicalsmiles:", info["canonicalsmiles"])
                    # print(drug_name, " isosmiles:", info["isosmiles"])
                    drug_smiles[drug_name] = info["canonicalsmiles"]

            if drug_name not in drug_smiles.keys():
                print(drug_name + " 没有找到smiles")
                continue

            if len(CID) >= 2:
                print(drug_name + " 有多个CID")

        except:  # noqa: E722
            print(json_data_smiles)
            print(json_data_cid)
            print(drug_name + " exception")
            exit()
    else:
        print(drug_name + " fuck")
        print(json_data_cid)
        exit()

    time.sleep(random.randint(0, 3))


with open("./datasets/preprocess/cppa/drug_smiles.pkl", "wb") as f:
    pickle.dump(drug_smiles, f)
