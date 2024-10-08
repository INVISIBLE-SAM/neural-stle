import altair as alt
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import PIL as Image

hub_handle='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

hub_module=hub.load(hub_handle)

def crop_image(img):
    corr_shape=img.shape
    new_shape=min(corr_shape[0],corr_shape[1]) #1000x600 it should take 600x600
    #considering optimal crop to be in center
    offset_y=max(corr_shape[0]-corr_shape[1],0)
    offset_y=max(corr_shape[1]-corr_shape[0],0)
    img=tf.image.corr_to_bounding_box(img,offset_y,offset_x,new_shape)
    return img

def load_image(uploaded_file,image_size=(256,256),col=st):
    img=Image.open(uploaded_file)
    img=tf.convert_to_tensor(img)#image to tensor
    img=crop_img(img) #adjusting image size if odd
    img=tf.image.resize(img,image_size)
    if img.shape[-1]==4:#if image is of 4d
        img=img[:,:,:3]
    
    img=tf.reshape(img,[-1,image_size[0],image[1],3])/255 # reshape and give size pixle normalize
    col.image(np.array(img))

def show_images(images,title=('',),col=st):
    n=len(images)
    for i in range(n):
        col.image(np.array(images[i][0]))
st.set_page_config(layout='wide')# content full width use
alt.renderers.set_embed_options(scaleFactor=2)#doubling the resolution of the charts, making them appear sharper and more detailed

if __name__=="__main__":
    img_width,img_height=384,384
    img_width_style,img_height_style=256,256

    col1,col2=st.columns(2)
    uploaded_file=col1.file_uploader("choose the image")
    if uploaded_file!=None:
        content_image = load_image(uploaded_file,(img_width,img_height),col=col1)

    uploaded_file=col2.file_uploader("choose the image")
    if uploaded_file!=None:
        style_image = load_image(uploaded_file,(img_width_style,img_height_style),col=col2)
        style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

        outputs=hub_module(tf.constant(content_image),tf.constant(style_image)) 
        stylized_image=outputs[0]#output bunch of image

        col3,col4,col5=st.columns(3)
        col4.markdown('# The Output')
        show_images([stylized_image],title=['stylized images'],col=col4)
