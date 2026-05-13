import os


def combine_paths(*paths):
    return os.path.join(*paths)


def list_files_in_folders(origin_folder_path, cassette_folder_path, telemetry_folder_path):
    cassette_files = os.listdir(
        origin_folder_path + '/' + cassette_folder_path)
    cassette_files.sort()
    telemetry_files = os.listdir(
        origin_folder_path + '/' + telemetry_folder_path)
    telemetry_files.sort()
    return cassette_files, telemetry_files


def get_pairs(files):
    pairs = {}
    for f in files:
        # Lấy 6 ký tự đầu (ví dụ: SC4001 hoặc ST7011)
        prefix = f[:6]

        if prefix not in pairs:
            pairs[prefix] = {}

        if "PSG" in f:
            pairs[prefix]["psg"] = f
        elif "Hypnogram" in f:
            pairs[prefix]["hypnogram"] = f

    # Chỉ trả về những ID có đủ cả 2 file
    return {k: v for k, v in pairs.items() if len(v) == 2}


def get_cassette_telemetry_pairs(origin_folder_path, cassette_folder_path, telemetry_folder_path):
    cassette_files, telemetry_files = list_files_in_folders(
        origin_folder_path, cassette_folder_path, telemetry_folder_path)
    cassette_pairs = get_pairs(cassette_files)
    telemetry_pairs = get_pairs(telemetry_files)

    return cassette_pairs, telemetry_pairs
