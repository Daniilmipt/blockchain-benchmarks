import subprocess
from bs4 import BeautifulSoup
import json

def start_network():

    # up containers
    result = subprocess.run('../networks/network.sh up createChannel -c bayesian-channel', shell=True, check=True)

    if result.returncode != 0:
        print("Error starting network")
        return result.returncode

    #deploy chaincode
    result = subprocess.run('../networks/network.sh deployCC -ccn fabcar -ccp ../caliper-benchmarks/src/fabric/samples/fabcar/go -ccl go -c bayesian-channel', shell=True, check=True)
    return result.returncode


def start_benchmark():
    result = subprocess.run('cd ../caliper-benchmarks &&' \
    'npx caliper launch manager --caliper-workspace ./ --caliper-networkconfig networks/fabric/test-network.yaml '
    '--caliper-benchconfig benchmarks/samples/fabric/fabcar/config.yaml --caliper-flow-only-test --caliper-fabric-gateway-enabled',
    shell=True, check=True)
    return result.returncode

def stop_network():
    result = subprocess.run('../networks/network.sh down', shell=True)
    return result.returncode

def transform_caliper_html_to_json(html_file, output_json_file):
    """
        Transform html report file to json format
    """
    try:
        with open(html_file, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
        
        summary = soup.find("div", {"id": "summary"}).text.strip() if soup.find("div", {"id": "summary"}) else "No summary found"
        metrics = {}
        
        table = soup.find("table")
        if table:
            headers = [th.text.strip() for th in table.find_all("th")]
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all("td")
                key = cells[0].text.strip()
                values = {headers[i]: cells[i].text.strip() for i in range(1, len(cells))}
                metrics[key] = values
        
        report_data = {
            "summary": summary,
            "metrics": metrics
        }
        
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump(report_data, json_file, indent=4)
        
        print(f"JSON report created: {output_json_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")


def calculate_average_tps(json_file):
    try:
        json_data = None
        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        tps_values = []
        for _, metrics in json_data["metrics"].items():
            tps_value = float(metrics.get("Throughput (TPS)", 0))
            tps_values.append(tps_value)
        
        if not tps_values:
            print("No TPS values found.")
            return None
        
        average_tps = sum(tps_values) / len(tps_values)
        return average_tps

    except Exception as e:
        print(f"Error: {e}")
        return None


# Каноничные ключи для TPS по раундам — стабильны между прогонами и удобны для CSV
ROUND_LABEL_TO_KEY = {
    "Create a car.":     "tps_create",
    "Change car owner.": "tps_change",
    "Query all cars.":   "tps_query_all",
    "Query a car.":      "tps_query_one",
}
ROUND_KEYS = list(ROUND_LABEL_TO_KEY.values()) + ["tps_avg"]


def calculate_round_tps(json_file) -> dict:
    """Парсит report.json и возвращает dict с TPS по каждому раунду + tps_avg.

    Возвращаемые ключи строго фиксированы: tps_create, tps_change,
    tps_query_all, tps_query_one, tps_avg. Если раунда нет в отчёте — NaN.
    """
    result = {k: float("nan") for k in ROUND_KEYS}
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        for label, key in ROUND_LABEL_TO_KEY.items():
            row = metrics.get(label)
            if row is None:
                continue
            tps_str = row.get("Throughput (TPS)")
            if tps_str is None:
                continue
            try:
                result[key] = float(tps_str)
            except ValueError:
                continue
        present = [v for v in (result[k] for k in ROUND_LABEL_TO_KEY.values()) if v == v]
        if present:
            result["tps_avg"] = sum(present) / len(present)
    except Exception as e:
        print(f"Error: {e}")
    return result


def observe_data() -> float:
    transform_caliper_html_to_json("../caliper-benchmarks/report.html", "../caliper-benchmarks/report.json")
    average_tps = calculate_average_tps("../caliper-benchmarks/report.json")
    return average_tps


def observe_round_tps() -> dict:
    """Как observe_data, но возвращает TPS по каждому раунду + среднее.

    Ключи: tps_create, tps_change, tps_query_all, tps_query_one, tps_avg.
    """
    transform_caliper_html_to_json(
        "../caliper-benchmarks/report.html",
        "../caliper-benchmarks/report.json",
    )
    return calculate_round_tps("../caliper-benchmarks/report.json")