# F.A.W.N
: Face Anonymization With Neural-networks

An selective face anonymization AI system that anonymizes all faces except for pre-registered individuals. This project extends the model [`face_anon_simple`](https://github.com/hanweikung/face_anon_simple) by adding
- ☑️ **Identity exclusion logic** for registered faces
- 📊 **Performace evaluation metircs** including FID, ReID, and AGR(AGR model: [`serengil_deepface`](https://github.com/serengil/deepface))
- 🌐 **Serverless web interface** built with Gradio

---
## 🔍 Project Goal
- Develop a **selective face anonymization system** that excludes user-specified individuals from anonymization.
- Enhance the anonymization quality and inference speed of the based model by **tuning hyperparameters**.
- Evaluate the anonymization quality of the tuned model using **multiple objective and perceptual metrics**.

## 🧪 Key Experiments
We conducted extensive experiments by adjusting the following hyperparmeters:
- `num_inference_steps` : number of diffusion steps (-> image quality and speed)
- `anonymization_degree` : degree of face transformation


## 📊 Evaluation Metrics
To evaluate the anonymization results from newly tuned model, we used:
- **Re-ID Rate** : Measure how accurately the generated image can be re-identified as the original one, using average cosine similarity. This follows a similar method to the one used in the reference paper.
- **FID(Frechet inception Distance)** : Assess the visual quality of anonymized images through the PyTorch FID module.
- **AGR Similarity** : Quantifies how naturally anonymized faces preserve Age, Gender and Race attributes by comparing them between the original and anonymized images.


## 💭 Results & Findings


## 🗒️ Sample Result Table


## 🧠 Insights

---
🛠️ **project for** : KSTA NIPA Google ML Bootcamp Project 


