import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
import os.path
import argparse
import cv2
import numpy as np
import _io
from tqdm import tqdm
import torch
from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate
#from session_state import get_state



def get_pair(latent_code,
                       boundary,
                        stepsize,
                       ):
  """Manipulates the given latent code with respect to a particular boundary
  and magnitude of manipulation.

  NOTE: Distance is sign sensitive.
"""
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])
  linspace = np.array([0., stepsize])
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                   f'W+ space in Style GAN!\n'
                   f'But {latent_code.shape} is received.')


## utility to handle latent code state
GAN_HASH_FUNCS = {
    _io.TextIOWrapper : id,
torch.nn.backends.thnn.THNNFunctionBackend:id,
torch.nn.parameter.Parameter:id,
torch.Tensor:id,
}

class LatentState:
    pass

@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,show_spinner=False)
def fetch_session(model,kwargs):
    session = LatentState()
    session.latent = init_latent(model,kwargs)
    return session

@st.cache(allow_output_mutation=True,suppress_st_warning=True,show_spinner=False)
def prepare_boudary(model_name, boundary_name, latent_space_type=None):
    boundary_load_state = st.text('Loading boundary...')
    basepath = './boundaries/'
    basepath= basepath+ model_name+'_'
    basepath = basepath + boundary_name + '_'
    if latent_space_type=='w':
        basepath+='w_'
    basepath+='boundary.npy'
    if not os.path.isfile(basepath):
        raise ValueError(f'Boundary `{basepath}` does not exist!')
    boundary = np.load(basepath)
    boundary_load_state.empty()
    return boundary


@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,suppress_st_warning=True,show_spinner=False)
def load_model(model_name, latent_space_type=None,logger=None):
    model_load_state = st.text('Loading GAN model...')
    gan_type = MODEL_POOL[model_name]['gan_type']
    if gan_type == 'pggan':
        model = PGGANGenerator(model_name, logger)
        kwargs = {}
    elif gan_type == 'stylegan':
        model = StyleGANGenerator(model_name, logger)
        kwargs = {'latent_space_type': latent_space_type}
    else:
        raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')
    print('loading',model_name)
    model_load_state.empty()
    return model,kwargs


## randomly initialize latent code
def init_latent(model,kwargs):
    return model.easy_sample(1, **kwargs)


## update the latent code from uploaded file
@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,suppress_st_warning=True,show_spinner=False)
def load_latent(latent_file, latent_state, model, kwargs):
    latent_codes = np.load(latent_file)
    latent_codes = model.preprocess(latent_codes, **kwargs)
    latent_state.latent = latent_codes
    return


@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS)
def get_logger(loggername):
    output_dir = './logs'
    logger = setup_logger(output_dir, logger_name=loggername)
    return logger


## set page to wide mode
def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

loggername = 'generate_data'
logger = get_logger(loggername)

st.title('InterfaceGAN Interactive Demo')

model_name = st.sidebar.selectbox(
            'Which GAN model?',
            ['pggan_celebahq','stylegan_celebahq','stylegan_ffhq'])

if model_name == 'stylegan':
    latent_space_type = model_name = st.sidebar.selectbox(
            'Which type of latent space?',
        ['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'])
else:
    latent_space_type = 'z'

gan_type = MODEL_POOL[model_name]['gan_type']
# Notify the reader that the data was successfully loaded.
model,kwargs = load_model(model_name, latent_space_type, logger)
inittemp = init_latent(model,kwargs)
latent_codes_state = fetch_session(model,kwargs)


stepsize = st.sidebar.slider('Manipulation step', -5.0, 5.0,0., 0.1)

boundaries_dict = {'pggan_celebahq':['age','age_c_eyeglasses','age_c_gender','age_c_gender_eyeglasses',
                            'eyegalsses','eyeglasses_c_age','eyeglasses_c_age_gender','eyeglasses_c_gender',
                            'gender','gender_c_age','gender_c_eyeglasses','gender_c_age_eyeglasses',
                            'pose','quality','smile'],
                   'stylegan_celebahq':['age',
                            'eyegalsses',
                            'gender',
                            'pose','smile'],
                   'stylegan_ffhq':['age',
                            'eyegalsses',
                            'gender',
                            'pose','smile']
                   }

boundary_name = st.sidebar.selectbox(
            'Which boundary?',
        boundaries_dict[model_name])

# Notify the reader that the data was successfully loaded.

boundary = prepare_boudary(model_name,boundary_name,latent_space_type)
#print(boundary.shape)

uploaded_boundary = st.sidebar.file_uploader("Or upload other boundaries", type="npy")
if uploaded_boundary is not None:
    boundary = np.load(uploaded_boundary)


num=1


if st.sidebar.button('Random face!'):
    latent_codes_state.latent = init_latent(model,kwargs)

uploaded_code = st.sidebar.file_uploader("Or upload a latent code .npz file", type="npy")
if uploaded_code is not None:
    load_latent(uploaded_code, latent_codes_state, model, kwargs)

latent_codes = latent_codes_state.latent
total_num = latent_codes.shape[0]
#print(latent_codes.shape)


face_edit_state = st.sidebar.text('Editing face...')
interpolations = get_pair(latent_codes,
                                    boundary,
                                    stepsize)

interpolation_id = 0
out_images = []
for interpolations_batch in model.get_batch_inputs(interpolations):
  if gan_type == 'pggan':
    outputs = model.easy_synthesize(interpolations_batch)
  elif gan_type == 'stylegan':
    outputs = model.easy_synthesize(interpolations_batch, **kwargs)
  for image in outputs['image']:
    out_images.append(image[:, :,:])
    interpolation_id += 1
st.image(out_images,width=512, caption=['Source face','Manipulated face'])
    #cv2.imwrite(save_path, image[:, :, ::-1])

#assert interpolation_id == args.steps
face_edit_state.text('Editing face...done!')