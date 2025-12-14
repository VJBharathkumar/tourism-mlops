from huggingface_hub import HfApi

HF_DATASET_REPO = "VJBharathkumar/tourism-wellness"

def main():
    api = HfApi()
    api.upload_file(
        path_or_fileobj="data/train.csv",
        path_in_repo="train.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj="data/test.csv",
        path_in_repo="test.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print("Uploaded train.csv and test.csv to HF dataset repo")

if __name__ == "__main__":
    main()