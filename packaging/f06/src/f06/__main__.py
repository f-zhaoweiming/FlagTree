import json
from . import doctor


def main():
    print(json.dumps(doctor(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
