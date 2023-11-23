import streamlit as st
from PIL import Image

from transformers import pipeline
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

st.set_page_config(layout='wide',
                   page_title='Garbage image classification'
                   )

def main():
    
    st.title("Garbage Classification")
    st.markdown("## Overview")   
    st.markdown("### Backgroud")  
    st.markdown("Garbage classification refers to the separation of several types of different categories in accordance with the environmental impact of the use of the value of the composition of garbage components and the requirements of existing treatment methods.")  
    st.markdown("The significance of garbage classification: ")  
    st.markdown("1. Garbage classification reduces the mutual pollution between different garbage, which is beneficial to the recycling of materials. ")  
    st.markdown("2. Garbage classification is conducive to reducing the final waste disposal volume. ")  
    st.markdown("3. Garbage classification is conducive to enhancing the degree of social civilization.")  
    st.markdown("### Dataset")  
    st.markdown("The garbage classification dataset is from Kaggle. There are totally 2467 pictures in this dataset. And this model is an image classification model for this dataset. There are 6 classes for this dataset, which are cardboard (393), glass (491), metal (400), paper(584), plastic (472), and trash(127).")  
    st.markdown("### Model")  
    st.markdown("The model is based on the [ViT](https://huggingface.co/google/vit-base-patch16-224-in21k) model, which is short for the Vision Transformer. It was introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), which was introduced in June 2021 by a team of researchers at Google Brain. And first released in [this repository](https://github.com/rwightman/pytorch-image-models). I trained this model with PyTorch. I think the most different thing between using the transformer to train on an image and on a text is in the tokenizing step. ")  
    st.markdown("There are 3 steps to tokenize the image:")  
    st.markdown("1. Split an image into a grid of sub-image patches")  
    st.markdown("2. Embed each patch with a linear projection")  
    st.markdown("3. Each embedded patch becomes a token, and the resulting sequence of embedded patches is the sequence you pass to the model.")  
    
    st.header("Try it out!")

    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
    
    if uploaded_file!=None:

        img=Image.open(uploaded_file)

        extractor = AutoFeatureExtractor.from_pretrained("yangy50/garbage-classification")
        model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")

        inputs = extractor(img,return_tensors="pt")
        outputs = model(**inputs)
        label_num=outputs.logits.softmax(1).argmax(1)
        label_num=label_num.item()

        st.write("The prediction class is:")

        if label_num==0:
            st.write("cardboard")
        elif label_num==1:
            st.write("glass")
        elif label_num==2:
            st.write("metal")
        elif label_num==3:
            st.write("paper")
        elif label_num==4:
            st.write("plastic")
        else:
            st.write("trash")

        st.image(img)


if __name__ == '__main__':
    main()
