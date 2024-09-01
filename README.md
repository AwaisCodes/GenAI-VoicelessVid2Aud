# GenAI-VoicelessVid2Aud

# Steps # 1
* Place your groq API key in either notebook file or python file first, to implement this code.

# Explanation
The code takes video file as input and the process through following steps:
 * It convert video in to list of frames
 * Remove background voice/noise
 * Caption frames into a list
 * Preprocessing using NLTK to remove stop words.
 * Create cluster of different captions and pick the unique ones to avoid redundencies.
 * Unique captions then fed to NLP model for summary.
 * Summary then fed to gtts for audio output.

 # Models used
 * cv2  - for video to frames
 * BLIP - salesforce blip model for image captioning
 * NLTK - for preprocessing of captions
 * K-means - using silhouette score for clusteringa on basis of uniqueness and sentense transformer LMmini - for avoiding redundancies
 * Llama 8b - for summarizing unique caption and explaning the whole scenerio
 * gTTs - from text to audio summary
