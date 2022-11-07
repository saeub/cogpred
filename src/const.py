from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / ".." / "data"
RAW_EEG_DATA_PATH = DATA_PATH / "raw" / "EEG"
PREPROCESSED_DATA_PATH = DATA_PATH / "preprocessed"
EEG_DATA_PATH = PREPROCESSED_DATA_PATH / "eeg"
PTA_DATA_PATH = PREPROCESSED_DATA_PATH / "measurements" / "PTA.csv"
COG_DATA_PATH = PREPROCESSED_DATA_PATH / "measurements" / "cognitive_measures.csv"

RANDOM_SEED = 42

HEARING_LOSS_THRESHOLD = 25.0

TFR_RESOLUTION = 50
EVENT_QUIET_START = 170
EVENT_QUIET_STOP = 240
CHANNELS = [
    "1-A1",
    "1-A2",
    "1-A3",
    "1-A4",
    "1-A5",
    "1-A6",
    "1-A7",
    "1-A8",
    "1-A9",
    "1-A10",
    "1-A11",
    "1-A12",
    "1-A13",
    "1-A14",
    "1-A15",
    "1-A16",
    "1-A17",
    "1-A18",
    "1-A19",
    "1-A20",
    "1-A21",
    "1-A22",
    "1-A23",
    "1-A24",
    "1-A25",
    "1-A26",
    "1-A27",
    "1-A28",
    "1-A29",
    "1-A30",
    "1-A31",
    "1-A32",
    "1-B1",
    "1-B2",
    "1-B3",
    "1-B4",
    "1-B5",
    "1-B6",
    "1-B7",
    "1-B8",
    "1-B9",
    "1-B10",
    "1-B11",
    "1-B12",
    "1-B13",
    "1-B14",
    "1-B15",
    "1-B16",
    "1-B17",
    "1-B18",
    "1-B19",
    "1-B20",
    "1-B21",
    "1-B22",
    "1-B23",
    "1-B24",
    "1-B25",
    "1-B26",
    "1-B27",
    "1-B28",
    "1-B29",
    "1-B30",
    "1-B31",
    "1-B32",
    "1-C1",
    "1-C2",
    "1-C3",
    "1-C4",
    "1-C5",
    "1-C6",
    "1-C7",
    "1-C8",
    "1-C9",
    "1-C10",
    "1-C11",
    "1-C12",
    "1-C13",
    "1-C14",
    "1-C15",
    "1-C16",
    "1-C17",
    "1-C18",
    "1-C19",
    "1-C20",
    "1-C21",
    "1-C22",
    "1-C23",
    "1-C24",
    "1-C25",
    "1-C26",
    "1-C27",
    "1-C28",
    "1-C29",
    "1-C30",
    "1-C31",
    "1-C32",
    "1-D1",
    "1-D2",
    "1-D3",
    "1-D4",
    "1-D5",
    "1-D6",
    "1-D7",
    "1-D8",
    "1-D9",
    "1-D10",
    "1-D11",
    "1-D12",
    "1-D13",
    "1-D14",
    "1-D15",
    "1-D16",
    "1-D17",
    "1-D18",
    "1-D19",
    "1-D20",
    "1-D21",
    "1-D22",
    "1-D23",
    "1-D24",
    "1-D25",
    "1-D26",
    "1-D27",
    "1-D28",
    "1-D29",
    "1-D30",
    "1-D31",
    "1-D32",
]

SUBJECT_IDS = [
    "0bsr6v",
    "0e5r4h",
    "0ntuoo",
    "0oyzpa",
    "0smfvj",
    "0v2pek",
    "10fghb",
    "10rz9l",
    "12ltfv",
    "1a479s",
    "1bcltx",
    "1cyzxq",
    "1gehfg",
    "1jcpga",
    "1n6aqp",
    "1ngprg",
    "1s1u1y",
    "1toogh",
    "1uzp40",
    "2cdflk",
    "2euds6",
    "2i56wv",
    "2icoaz",
    "2iif87",
    "2nq9f4",
    "2oiode",
    "2qn10p",
    "2t7k11",
    "2tku8e",
    "30gmga",
    "32cosp",
    "35doj2",
    "3abgx0",
    "3gdmda",
    "3ipb68",
    "3izwvq",
    "3jwrm3",
    "3kduyt",
    "3kperz",
    "3oz8tm",
    "3pdegp",
    "3s9x2q",
    "3tdfx3",
    "3thqp8",
    "3ucbt4",
    "3xof4b",
    "3zk63p",
    "40i7j1",
    "47tuxc",
    "47y10c",
    "48cquy",
    "4j3i7m",
    "4loa2",
    "4n12gr",
    "4p1g59",
    "4rc5cp",
    "4sdj2b",
    "4sg7vp",
    "4smdfr",
    "4tzxge",
    "4w2ugi",
    # "57yhxc",  # PTA missing
    "5brg9a",
    "5cyshu",
    "5j390m",
    "5j6vsw",
    "5jle8y",
    "5kyxeo",
    "5o1o31",
    "5on9c8",
    "5rm6ne",
    "5utf4w",
    "5xz1jy",
    "5yzckj",
    "5z4vul",
    "5zdxr2",
    "62djma",
    "66vl4p",
    "68tefj",
    "6btzgm",
    "6ddsfu",
    "6dnhrm",
    "6ejovj",
    "6fqh62",
    "6g6ks",
    "6if3t9",
    "6kpsr6",
    "6n4olv",
    "6pgjfe",
    "6qer45",
    "6ru40v",
    "6s32xn",
    "6v7hrn",
    "71rpqm",
    "78zjhw",
    "7d653j",
    "7euqan",
    "7euqao",
    "7fth2e",
    "7lpg69",
    "7pd8a1",
    "7sw8iy",
    "7tp9ca",
    "7vhkqi",
    "7xq4tt",
    "7xtxtz",
    "7xxyw8",
    "7z5zaz",
    "7zgbd4",
    "8aplfg",
    "8i1jsz",
    "8inukl",
    "8k6oay",
    "8niu5o",
    "8p81ay",
    "8qbi4a",
    "8qn1x2",
    "8tsirw",
    "8xy0qd",
    "8y2zdo",
    "8zrfmv",
    "8zzmsu",
    "9abc5e",
    "9ckv8g",
    "9fer8n",
    "9gnw2c",
    "9gzbkc",
    "9pyjap",
    "9qcybd",
    "9shccd",
    "9u9tsl",
    "9yvccf",
    "ctc7u",
]
