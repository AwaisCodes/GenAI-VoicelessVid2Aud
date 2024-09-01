
"""

# Install necessary packages
!pip install opencv-python-headless pillow transformers gtts nltk sentence-transformers scikit-learn groq
!apt-get install -y ffmpeg

# Import libraries
import cv2
import os
import numpy as np
from google.colab import files
from transformers import BlipProcessor, BlipForConditionalGeneration
import PIL.Image
from gtts import gTTS
from IPython.display import Audio, display
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from groq import Groq

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set environment variable for Groq API key
os.environ['GROQ_API_KEY'] = 'place_your_groq_api_key_here'  # Replace with your actual Groq API key

# Upload the video file
uploaded = files.upload()
video_path = next(iter(uploaded))

# Create a directory to store frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Calculate the total number of frames
def get_total_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

# Extract frames at equal intervals
def extract_frames(video_path, output_folder, percentage=30):
    total_frames = get_total_frames(video_path)
    selected_frames = int(total_frames * percentage / 100)
    interval = total_frames / selected_frames
    video = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % int(interval) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            count += 1
        frame_count += 1
    video.release()

extract_frames(video_path, 'frames', percentage=30)

# Load the BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    try:
        image = PIL.Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return None

def gather_captions(folder_path):
    captions = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            caption = generate_caption(image_path)
            if caption:
                captions.append(caption)
    return captions

# Preprocess function
def preprocess_captions(captions):
    def preprocess_caption(caption):
        caption = caption.lower()
        tokens = word_tokenize(caption)
        tokens = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    return [preprocess_caption(caption) for caption in captions]

# Find the optimal number of clusters using silhouette score
def find_optimal_clusters(embeddings):
    possible_clusters = range(2, 21)  # Test from 2 to 20 clusters
    best_n_clusters = 2
    best_score = -1

    for n_clusters in possible_clusters:
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(embeddings)
        labels = kmeans.labels_
        if len(set(labels)) > 1:  # Avoid silhouette score calculation with a single cluster
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

    return best_n_clusters

# Remove redundant captions using optimized KMeans clustering
def remove_redundancies(captions):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(captions)
    print("Caption Embeddings:")
    print(embeddings)  # Debug: Print embeddings

    # Determine optimal number of clusters
    optimal_clusters = find_optimal_clusters(embeddings)
    print(f"Optimal Number of Clusters: {optimal_clusters}")

    # Perform KMeans clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, n_init='auto', random_state=0).fit(embeddings)
    labels = kmeans.labels_
    print("KMeans Labels:")
    print(labels)  # Debug: Print KMeans labels

    # Select one caption per cluster
    unique_captions = []
    for i in range(optimal_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            unique_captions.append(captions[cluster_indices[0]])

    print(f"Unique Captions after clustering: {unique_captions}")
    return unique_captions

# Generate summary with Llama 8B using Groq API
def generate_summary_with_llama(captions):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    text = ' '.join(captions)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Summarize the scenerio without redundancy, into minimum two lines/para: {text}",
            }
        ],
        model="llama3-8b-8192",
    )
    summary = chat_completion.choices[0].message.content
    return summary

# Process captions and generate final summary
captions = gather_captions('frames')

# Display captions
print("Frame Captions:")
for caption in captions:
    print(caption)

# Preprocess and remove redundancies
preprocessed_captions = preprocess_captions(captions)
print("Preprocessed Captions:")
print(preprocessed_captions)  # Debug: Print preprocessed captions

unique_captions = remove_redundancies(preprocessed_captions)

# Generate and display the final summary
if unique_captions:
    final_summary = generate_summary_with_llama(unique_captions)
    print(f"Total frames processed: {len(os.listdir('frames'))}")
    #print("Final Summary of the video:")
    print(final_summary)
else:
    print("No unique captions found.")

# Convert the final summary to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = '/content/output.mp3'
    tts.save(audio_file)
    return audio_file

# Create and play the audio file
audio_file = text_to_speech(final_summary)
display(Audio(audio_file, autoplay=True))