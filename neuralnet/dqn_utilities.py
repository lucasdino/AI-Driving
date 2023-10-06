import torch
import numpy as np
import csv
import datetime
import pygame
    

def instantiate_hardware(max_episodes):
    """Function to load CUDA if GPU is available; otherwise use CPU""" 
    hardware = "tbu"
    num_episodes = 0
    if torch.cuda.is_available():
        hardware = "cuda"
        num_episodes = max_episodes
    else:
        hardware = "cpu"
        num_episodes = max_episodes // 100

    print(f"Utilizing {hardware} for neural net hardware.")
    return torch.device(hardware), num_episodes
 

def random_action_motion(action_space_size, last_action):
    """Function provides random motion to car"""
    rand = np.random.random()
    action_list = [0]*action_space_size
    action_index = 0

    if last_action == None:        
        action_index = np.random.randint(action_space_size-2)
    else:
        prev_action_index = last_action.index(1)

        # Bias toward having car do the same action as before. Prevents super wobbly behavior. Otherwise add 1 / subtract 1 (min/max to prevent OOB)
        if rand < 0.2:
            action_index = prev_action_index
        elif rand < 0.6:
            action_index = max(prev_action_index-1, 0)
        else:
            action_index = min(prev_action_index+1, action_space_size-1)
        
    action_list[action_index] = 1
    return action_list
    

def convert_action_tensor_to_list(action_tensor, action_space_size):
    """Model that takes in an action tensor and returns a list that can be interpreted by the game"""
    inferred_action = [0] * action_space_size
    inferred_action[action_tensor.item()] = 1
    return inferred_action


def print_progress(progress_vars, loss_memory, max_episodes, cur_time):
    """Function to print training progress to the console every 'print_freq' episodes"""
    loss_memory_tensor = torch.tensor(loss_memory, dtype=torch.float)

    if (progress_vars["Episodes"] % progress_vars["PrintFrequency"] == 0) or (progress_vars["Episodes"] == max_episodes-1):
        # Compute performance calculations over the desired window
        avg_loss = loss_memory_tensor[progress_vars["IndexOfLastPrint"]:].mean().item()
        avg_score = round(sum(progress_vars["EpisodeTotalReward"][-progress_vars["PrintFrequency"]:])/progress_vars["PrintFrequency"], 0)
        fps = round((len(loss_memory_tensor) - progress_vars["IndexOfLastPrint"]) / ((cur_time - progress_vars["TimeAtLastPrint"])/1000), 0)
        
        # Print statement 
        print(f"Ep: {progress_vars['Episodes']}/{max_episodes}; Avg Loss: {avg_loss:.2f} ({progress_vars['IndexOfLastPrint']}-{len(loss_memory_tensor)}); Ep. Coins: {progress_vars['CoinsSinceLastPrint']}; Avg End Score: {avg_score}; FPS: {fps}")
        
        # Reset progress variables
        progress_vars["TimeAtLastPrint"] = cur_time
        progress_vars["CoinsSinceLastPrint"] = 0
        progress_vars["IndexOfLastPrint"] = len(loss_memory_tensor)


def save_loss_to_csv(loss_calculations, filename):
    """Saves the loss to a CSV file."""
    with open(f'assets/loss_results/{filename}.csv', "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for loss in loss_calculations:
            csv_writer.writerow([loss])
        

def export_params_and_loss(loss_memory, policy_net, target_net):
    """Function to export neural net weights and loss"""
    formatted_datetime = datetime.datetime.now().strftime("%m.%d.%y-%H.%M")
    save_loss_to_csv(loss_memory, "Loss_Calculations-" + formatted_datetime)
    policy_net.export_parameters("Policy_Net_Params-" + formatted_datetime)
    target_net.export_parameters("Target_Net_Params-" + formatted_datetime)


def check_toggles(model_toggles, model_toggle_buttons):    
    for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit()
            if ((event.type == pygame.MOUSEBUTTONDOWN) and model_toggles["ClickEligible"]):
                model_toggles["ClickEligible"] = False
                if model_toggle_buttons["toggle_render_button"].collidepoint(event.pos):
                    model_toggles["Render"] = not model_toggles["Render"]
                if model_toggle_buttons["export_weights_button"].collidepoint(event.pos):
                    model_toggles["SaveWeights"] = True
            if (event.type == pygame.MOUSEBUTTONUP):
                model_toggles["ClickEligible"] = True
    return model_toggles