import os

if __name__ == "__main__":
    print(f"Current dir's folders/files: {os.listdir('./')}")
    print(f"Previous dir's folders/files: {os.listdir('../')}")
    print(f"One level up's folders/files: {os.listdir('../../')}")
