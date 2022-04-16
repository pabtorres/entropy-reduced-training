import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from persist_results import *

import torchvision
import torchvision.transforms as transforms
from dataset_classes_instantiate import *

# Herramientas de entrenamiento
from herrramientas_de_entrenamiento_i9_gray import train_for_classification

from checkpoint_utils_v2 import *

# Importar interpolador
from interpolacion_lineal import interpolacion

# Importar generador de transformaciones
from generador_de_transformaciones_v2_gray import make_transformation_selection

# Redes neuronales
from networks_controller_v13_e import instantiate_network

# Prueba conjuntos
from prueba_sobre_conjuntos_v3 import test_over_sets

# Generador csv resultados por clase
from resultados_por_clase_csv import results_per_class


# DEVICE
DEVICE = 'cuda'

def funcion_corrige_name(value):
  if value<10: return f'0{value}'
  return value

# Config seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Early Stopping
early_stopping_num = 3


def run_it(dimension, delta_alfa, red, input_folder):
  # FILE SYSTEM
  # [STAR] IMPORTANTE CREAR LAS CARPETAS EN EL SERVIDOR ANTES
  #EXPERIMENT_MIX = f'{"80c0r"}' # <--- [STAR] Nombre experimento <- TUNE/ADJUST HERE !
  DIMENSION = dimension
  
  print(f'Dimension is: {dimension}')
  print(f'Valor de alfa is: {delta_alfa}')
  print(f'Red is: {red}')

  D_ALFA = f'DeltaAlfa_0{delta_alfa[2:]}'

  letter_variant = input_folder

  input_folder = f'F_PairedEpochs_Fixed_{input_folder}'

  NEURAL_NETWORK_NAME = red # [STAR] Tiene una ocurrencia abajo, en caso de cambiarla, cambiar ambas
  FOLDER_EXPERIMENT = f'New_PairedEpochs_Fixed/{input_folder}/{NEURAL_NETWORK_NAME}/{DIMENSION}/{D_ALFA}'
  FOLDER_REPORTS = f'{FOLDER_EXPERIMENT}/Reports'
  FOLDER_TEST_REPORTS = f'{FOLDER_EXPERIMENT}/TestReports'
  FOLDER_CHECKPOINTS = f'{FOLDER_EXPERIMENT}/Checkpoints'
  FOLDER_MEPIS_REDUCTION = f'{FOLDER_EXPERIMENT}/MEPIS'
  FOLDER_TRAINING_REPORTS = f'{FOLDER_EXPERIMENT}/TrainingReports'
  #ANGULAR_STONE_CHECKPOINT = f'ExperimentosAlpha/{NEURAL_NETWORK_NAME}/{"AngularStoneCheckpoint"}/AngularStoneFor-{NEURAL_NETWORK_NAME}'
  #FILE_FOLDER = f'ResultadosTestingExperimentosAlphaRemake2/{NEURAL_NETWORK_NAME}/{DIMENSION}/{EXPERIMENT_MIX}'

  transform = transforms.Compose(
      [transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),])

  print(f'Training')
  # Definamos algunos hiper-parámetros
  BATCH_SIZE = 32
  EPOCHS = 80
  REPORTS_EVERY = 1
  CICLOS = 80

  # Wrapper red
  wrapper_net = instantiate_network(red)

  # Actualizar puntos de control, accuracyes, etc
  wrapper_net.update(letter_variant)

  # Instanciar red neuronal
  net = wrapper_net.get_net()

  # Hiperparametros de la red
  LR = wrapper_net.get_lr()
  optimizer = wrapper_net.get_optimizer()
  criterion = nn.CrossEntropyLoss()
  scheduler = wrapper_net.get_scheduler()

  # Checkpoint
  print(f'Cargando pesos de punto de control')
  print(wrapper_net.get_checkpoint())
  checkpoint = torch.load(wrapper_net.get_checkpoint())
  net.load_state_dict(checkpoint['model_state_dict'])

  print(f'Load optimizer')
  load_optimizer_checkpoint_cuda(optimizer, checkpoint)

  print(f'Load learning rate')
  for g in optimizer.param_groups:
    g['lr'] = wrapper_net.get_checkpoint_lr()

  print(f'Loading scheduler')
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  # Dataloaders
  train_loader = DataLoader(TrainingDataset(transform=transform, list='humanet_training_set_v3_f.json'), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=8)
  test_loader = DataLoader(ValidationDataset(transform=transform, list='humanet_minival_set_v3_f.json'), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)
  test_set_loader = DataLoader(ValidationDataset(transform=transform, list='testing_tesis_categories_20_class.json'), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

  # Inicialización variables reducción
  d_alpha = float(delta_alfa)
  DIMENSION = dimension

  # Acc
  # Cada red entrega ese valor
  acc_val_actual = wrapper_net.get_checkpoint_acc()
  acc_val_anterior = wrapper_net.get_checkpoint_acc()

  print(f'Reportar acc {acc_val_anterior}')

  # Lista resultados
  loss_reports = []
  acc_train = []
  acc_valid = []
  epoch_time = []
  acc_valid_2 = []
  acc_color = []
  acc_combined = []
  acc_crop = []
  acc_resolution = []
  acc_complete = []

  # Inicio época clásica, reducida, round y creacion checkpoint
  checkpoint_creation = []
  begin_round = []
  alpha_of_round = []
  checkpoint_load = []

  # Lista de resultados sin valores omitidos
  loss_reports_o = []
  acc_train_o = []
  acc_valid_o = []
  epoch_time_o = []
  acc_valid_2_o = []
  acc_color_o = []
  acc_combined_o = []
  acc_crop_o = []
  acc_resolution_o = []
  alfa_values_o = []

  # Almacenar tipo de época
  tipo_clasica = []
  tipo_reducida = []

  # Inicialización variables early stopping
  early_stopper_watcher = 0

  # Inicialización alfa actual
  alpha_actual = 1

  # Inicialización checkpoint
  mejor_modelo = wrapper_net.get_checkpoint()

  # Inicialización lr checkpoint
  mejor_modelo_lr = wrapper_net.get_checkpoint_lr()

  epoca_actual = 1
  for i in range(CICLOS):

    # Almacenar inicio round
    begin_round.append(epoca_actual)


    # Reinicialización variables early stopping
    early_stopper_pair = 0

    alpha_actual -= d_alpha

    # Antes de crear una transformación validar valor de alfa actual
    if alpha_actual <= 0.0:
      alpha_actual = 1.0 - d_alpha
    
    # Almacenar valor de alfa actual
    alpha_of_round.append(alpha_actual)
    
    reducir = True
    reduccion = make_transformation_selection(alpha_actual, selection=DIMENSION)


    if early_stopper_watcher == early_stopping_num:
      # No se ejecutó la nueva época por lo que la época actual se resta 1-
      print(f'Early Stopped in epoch: {epoca_actual-1}')
      break

    for idx_classic in range(EPOCHS):
      flag_augment_stopper = True
    # Epocas Reducidas
      #print("TRAINING REDUCED")
      print(f'r: {alpha_actual}')
      tipo_reducida.append(1)
      tipo_clasica.append(0)
      train_loss, acc, e_time = train_for_classification(net, train_loader, 
                                                test_loader, optimizer, 
                                                criterion, lr_scheduler=scheduler, 
                                                epochs=1, reports_every=REPORTS_EVERY, device='cuda',
                                                reducir=reducir, reduccion=reduccion)
      #print("Almacenando listas")
      persist_tuple_prefix((train_loss, acc), f'{FOLDER_REPORTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC_REPORT')

      # Actualizar listas de resultados
      acc_train.append(acc[0][0])
      acc_valid.append(acc[1][0])
      loss_reports.append(train_loss[0])
      epoch_time.append(e_time)

      # Actualizar las otras listas también
      lista_de_accuracys = test_over_sets(net)
      acc_valid_2.append(lista_de_accuracys[0])
      acc_color.append(lista_de_accuracys[1])
      acc_combined.append(lista_de_accuracys[2])
      acc_crop.append(lista_de_accuracys[3])
      acc_resolution.append(lista_de_accuracys[4])
      acc_complete.append(lista_de_accuracys[5])

      #print(acc_valid)
      #print(acc_valid_2)
      #print(acc_combined)

      if acc[1][0] > acc_val_actual:
        flag_augment_stopper = False
        early_stopper_watcher = 0
        early_stopper_pair = 0
        acc_val_actual = acc[1][0]
        print(f'Valor de acc_val_actual actualizado a {acc_val_actual}, almacenando punto de control')
        # Almacenar punto de control época clásica
        PATH = f'./{FOLDER_CHECKPOINTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC'
        save_checkpoint_2(net, optimizer, epoca_actual, PATH, scheduler)
        mejor_modelo = PATH
        mejor_modelo_lr = scheduler.get_last_lr()[0]
        checkpoint_creation.append(epoca_actual)


      epoca_actual+=1

    # Épocas Clásicas
      #print("TRAINING CLASSIC")
      print(f'c: {alpha_actual}')
      tipo_reducida.append(0)
      tipo_clasica.append(1)
      train_loss, acc, e_time = train_for_classification(net, train_loader, 
                                                test_loader, optimizer, 
                                                criterion, lr_scheduler=scheduler, 
                                                epochs=1, reports_every=REPORTS_EVERY, device='cuda',
                                                reducir=False, reduccion=None)
      #print("Almacenando listas")
      persist_tuple_prefix((train_loss, acc), f'{FOLDER_REPORTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC_REPORT')

      # Actualizar listas de resultados
      acc_train.append(acc[0][0])
      acc_valid.append(acc[1][0])
      loss_reports.append(train_loss[0])
      epoch_time.append(e_time)

      # Actualizar las otras listas también
      lista_de_accuracys = test_over_sets(net)
      acc_valid_2.append(lista_de_accuracys[0])
      acc_color.append(lista_de_accuracys[1])
      acc_combined.append(lista_de_accuracys[2])
      acc_crop.append(lista_de_accuracys[3])
      acc_resolution.append(lista_de_accuracys[4])
      acc_complete.append(lista_de_accuracys[5])

      #print(acc_valid)
      #print(acc_valid_2)
      #print(acc_combined)


      if acc[1][0] > acc_val_actual:
        flag_augment_stopper = False
        early_stopper_watcher = 0
        early_stopper_pair = 0
        acc_val_actual = acc[1][0]
        print(f'Valor de acc_val_actual actualizado a {acc_val_actual}, almacenando punto de control')
        # Almacenar punto de control época clásica
        PATH = f'./{FOLDER_CHECKPOINTS}/ImageNet-{NEURAL_NETWORK_NAME}-{funcion_corrige_name(epoca_actual)}_CHECKPOINT_CLASSIC'
        save_checkpoint_2(net, optimizer, epoca_actual, PATH, scheduler)
        mejor_modelo = PATH
        mejor_modelo_lr = scheduler.get_last_lr()[0]
        checkpoint_creation.append(epoca_actual)

      # Actualizar watcher
      if flag_augment_stopper:
        early_stopper_pair += 1

      if not flag_augment_stopper:
        # Actualizar listas de resultados
        acc_train_o += acc_train[-2:]
        acc_valid_o += acc_valid[-2:]
        loss_reports_o += loss_reports[-2:]
        epoch_time_o += epoch_time[-2:]

        # Actualizar las otras listas también
        acc_valid_2_o += acc_valid_2[-2:]
        acc_color_o += acc_color[-2:]
        acc_combined_o += acc_combined[-2:]
        acc_crop_o += acc_crop[-2:]
        acc_resolution_o += acc_resolution[-2:]

        # Actualizar lista alfas
        alfa_values_o += [alpha_actual, alpha_actual]
      
      if early_stopper_pair == early_stopping_num:
        print(f'Early Stopped in epoch: {epoca_actual}')
        epoca_actual+=1 # Aumentar valor para la época reducida siguiente
        break


      epoca_actual+=1

    # Si no hubo cambios en el accuracy aumenta el contador
    if acc_val_anterior == acc_val_actual:
      # Buscar checkpoint de mejor modelo.
      print(f'Best checkpoint is in: {mejor_modelo}')
      checkpoint = torch.load(mejor_modelo)
      net.load_state_dict(checkpoint['model_state_dict'])
      load_optimizer_checkpoint_cuda(optimizer, checkpoint)
      for g in optimizer.param_groups:
        g['lr'] = mejor_modelo_lr
      print(f'Load scheduler')
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      early_stopper_watcher += 1
      checkpoint_load.append(epoca_actual)

    # Si se actualizó el valor entonces se reinicia el contador y actualizar acc_val_anterior
    else:
      early_stopper_watcher = 0
      acc_val_anterior = acc_val_actual

  
  # Almacenar listas
  persist_list(acc_train, f'{FOLDER_TRAINING_REPORTS}/TrainingAccuracy')
  persist_list(acc_valid, f'{FOLDER_TRAINING_REPORTS}/ValidationAccuracy')
  persist_list(loss_reports, f'{FOLDER_TRAINING_REPORTS}/TrainingLoss')
  persist_list(epoch_time, f'{FOLDER_TRAINING_REPORTS}/EpochsTime')

  # Almacenar las otras listas también
  persist_list(acc_color, f'{FOLDER_TRAINING_REPORTS}/TestColor')
  persist_list(acc_combined, f'{FOLDER_TRAINING_REPORTS}/TestCombined')
  persist_list(acc_crop, f'{FOLDER_TRAINING_REPORTS}/TestCrop')
  persist_list(acc_resolution, f'{FOLDER_TRAINING_REPORTS}/TestResolution')
  persist_list(acc_complete, f'{FOLDER_TRAINING_REPORTS}/TestComplete')


  # Almacenar valores saltando valores no usados
  # Alfa values
  persist_list(alfa_values_o, f'{FOLDER_TRAINING_REPORTS}/OAlfaValues')

  # Almacenar listas
  persist_list(acc_train_o, f'{FOLDER_TRAINING_REPORTS}/OTrainingAccuracy')
  persist_list(acc_valid_o, f'{FOLDER_TRAINING_REPORTS}/OValidationAccuracy')
  persist_list(loss_reports_o, f'{FOLDER_TRAINING_REPORTS}/OTrainingLoss')
  persist_list(epoch_time_o, f'{FOLDER_TRAINING_REPORTS}/OEpochsTime')

  # Almacenar las otras listas también
  persist_list(acc_color_o, f'{FOLDER_TRAINING_REPORTS}/OTestColor')
  persist_list(acc_combined_o, f'{FOLDER_TRAINING_REPORTS}/OTestCombined')
  persist_list(acc_crop_o, f'{FOLDER_TRAINING_REPORTS}/OTestCrop')
  persist_list(acc_resolution_o, f'{FOLDER_TRAINING_REPORTS}/OTestResolution')

  # Almacenar listas
  persist_list(checkpoint_creation, f'{FOLDER_TRAINING_REPORTS}/CheckpointCreation')
  persist_list(begin_round, f'{FOLDER_TRAINING_REPORTS}/BeginRoundEpoch')
  persist_list(alpha_of_round, f'{FOLDER_TRAINING_REPORTS}/AlphaOfTheRound')
  persist_list(checkpoint_load, f'{FOLDER_TRAINING_REPORTS}/CheckpointLoads')

  # Almacenar lista tipo de epoca
  persist_list(tipo_clasica, f'{FOLDER_TRAINING_REPORTS}/ListaTipoDeEpocaClasica')
  persist_list(tipo_reducida, f'{FOLDER_TRAINING_REPORTS}/ListaTipoDeEpocaReducida')

  # Almacenar csv con resultados finales
  results_per_class(net, FOLDER_TRAINING_REPORTS)


if __name__ == '__main__':
    parameter_list_names = ["Dimension", "Valor de Alfa", "Red Neuronal", "Carpeta"]
    sys_arguments = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    for p_l_n, sys_arg in zip(parameter_list_names, sys_arguments):
      print(f'Value for {p_l_n} is {sys_arg}')
    run_it(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])