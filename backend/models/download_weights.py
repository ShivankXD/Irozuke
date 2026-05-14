from huggingface_hub import hf_hub_download

# Download tiny first to test (220MB)
hf_hub_download(
    repo_id="piddnad/DDColor-models",
    filename="ddcolor_artistic.pth",
    local_dir="."   # saves into backend/models/
)

print("Downloaded! Now test inference before downloading the 912MB artistic model.")