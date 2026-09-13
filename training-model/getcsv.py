import kagglehub

# Download latest version
path = kagglehub.dataset_download("canerkonuk/youtube-trending-videos-global")

print("Path to dataset files:", path)