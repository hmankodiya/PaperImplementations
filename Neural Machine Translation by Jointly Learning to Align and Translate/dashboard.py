import streamlit as st
import utils as ut

PARAMS = './params.yml'
st.title('Enter parameters here')

if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False

run_name = st.text_input('run name goes here', 
                         disabled=st.session_state.disabled)
hidden_size = int(st.number_input('input hidden size', max_value=1000,
                                  disabled=st.session_state.disabled))
epochs = int(st.number_input('epochs', max_value=20, 
             disabled=st.session_state.disabled))
learning_rate = st.number_input('learning rate',
                                disabled=st.session_state.disabled)
test_split = st.number_input('test split', 
                             disabled=st.session_state.disabled)
num_workers = int(st.number_input('total cpu workers', 
                                  disabled=st.session_state.disabled))

parameters = {'model_parameters': {'run_name': run_name,
                                 'hidden_size': hidden_size, 
                                 'epochs': epochs, 
                                 'learning_rate':learning_rate,
                                 'test_split': test_split,
                                 'num_workers': num_workers}
              }

if st.button('Save Parameters'):
    with st.spinner('Saving parameters'):
        ut.write_yaml(PARAMS, parameters)
    st.success(f'Parameters saved succesfully at {PARAMS}')
