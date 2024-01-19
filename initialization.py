import numpy as np
import pandas as pd
import pytorch


def load_data(node_file, edge_file):
    
    # Load data from files
    df_node = pd.read_csv(node_file, delimiter=" ")
    df_edge = pd.read_csv(edge_file, delimiter=" ")

    x_init, y_init, theta_init = df_node['X'].to_numpy(), df_node['Y'].to_numpy(), df_node['theta'].to_numpy()
    
    # IDout IDin dx dy dth
    dx_measurement = df_edge[df_edge.columns[1:6]].to_numpy()
    
    # Construct Information Matrix
    info_matrix=np.zeros((3,3,dx_measurement.shape[0]))
    info_matrix[0,0,:]=df_edge['I11'].to_numpy()
    info_matrix[1,1,:]=df_edge['I22'].to_numpy()
    info_matrix[2,2,:]=df_edge['I33'].to_numpy()
    info_matrix[0,1,:]=info_matrix[1,0,:]=df_edge['I12'].to_numpy()
    info_matrix[0,2,:]=info_matrix[2,0,:]=df_edge['I13'].to_numpy()
    info_matrix[1,2,:]=info_matrix[2,1,:]=df_edge['I23'].to_numpy()

    return np.vstack((x_init.reshape(1,-1), y_init.reshape(1,-1), theta_init.reshape(1,-1))), dx_measurement, info_matrix


def calc_rot(theta):

    #Calculate rotation matrix
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def diff_rot(theta):

    #Calculate differentiated rotation matrix
    return np.array([[-np.sin(theta), np.cos(theta)], [-np.cos(theta), -np.sin(theta)]])

def calc_jacobian(state_i, state_j):

    j_i = np.zeros((3,3))
    j_i[0:2,0:2] = np.matmul(calc_rot(state_i[2]), np.array([[state_j[0]-1, 0],[0, state_j[1]-1]]))
    j_i[0:2,2] = np.matmul(diff_rot(state_i[2]),state_j[:2].reshape(-1,1) - state_i[:2].reshape(-1,1))
    j_i[2,2] = -1

    j_j = np.zeros((3,3))
    j_j[0:2,0:2] = np.matmul(calc_rot(state_i[2]), np.array([[1-state_i[0], 0],[0, 1-state_i[1]]]))
    j_j[2,2] = 1

    return j_i, j_j