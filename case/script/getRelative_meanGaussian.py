import shutil
import pandas as pd
import glob
import os

def copy_file(source_path, destination_path):
    try:
        shutil.copyfile(source_path, destination_path)
        print(f"成功将 {source_path} 复制到 {destination_path}")
    except FileNotFoundError:
        print("错误：源文件不存在！")
    except PermissionError:
        print("错误：没有写入目标文件的权限！")
    except Exception as e:
        print(f"发生未知错误: {e}")

def copy_file_to_dir(source_path, destination_path):
    try:
        os.makedirs(destination_path, exist_ok=True)
        shutil.copy(source_path, destination_path)
        print(f"成功将 {source_path} 复制到 {destination_path}")
    except FileNotFoundError:
        print("错误：源文件不存在！")
    except PermissionError:
        print("错误：没有写入目标文件的权限！")
    except Exception as e:
        print(f"发生未知错误: {e}")



def get_relative_meanGaussian():
    mutationList = pd.read_csv('../data/mutationName.csv')['mutation'].tolist()

    for mut in mutationList:
        file_A = f"/home/data1/BGM/mdrugEffect/bigData/dockEnergyFilterModify/Abl1_{mut}_relative_meanGaussian.csv"
        file_B = f"../data/dockEnergyFilterModify/Abl1_{mut}_relative_meanGaussian.csv"


        copy_file(file_A, file_B)

def get_exp_ligand():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element1'].tolist()
    for mut in mutationList:
        file_A = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/docked/{}_*/{}_*_{}_*_vina_output.mol2'.format(mut.split('_')[0], mut.split('_')[0], mut.split('_')[1]))[0]
        file_B = f"../data/exp/autoDockScript_small/docked/{mut.split('_')[0]}_copy/"
        copy_file_to_dir(file_A, file_B)

# get_exp_ligand()

def get_exp_protein_pdbqt():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element1'].tolist()
    for mut in mutationList:
        file_A = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*_pocket.pdbqt'.format(mut.split('_')[0], mut.split('_')[0]))[0]
        file_B = f"../data/exp/autoDockScript_small/aligned/{mut.split('_')[0]}_copy/"
        copy_file_to_dir(file_A, file_B)

# get_exp_protein_pdbqt()

def get_exp_protein_pdb():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element1'].tolist()
    for mut in mutationList:
        file_A = glob.glob('/home/data1/BGM/mutationEffect/autoDockScript_small/aligned/{}_*/{}_*_pocket.pdb'.format(mut.split('_')[0], mut.split('_')[0]))[0]
        file_B = f"../data/exp/autoDockScript_small/aligned/{mut.split('_')[0]}_copy/"
        copy_file_to_dir(file_A, file_B)

get_exp_protein_pdb()

def get_af_ligand_pdbqt():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element2'].tolist()
    for mut in mutationList:
        afKey = mut
        afMut = afKey.split('_')[0]
        modi = afKey.split('_')[1]
        ri = afKey.split('_')[2]
        seedi = afKey.split('_')[3]
        afKpi = afKey.split('_')[4]
        file_A = glob.glob('/home/data1/BGM/mutationEffect/ensemble/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out.pdbqt'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
        file_B = f"../data/af/autodock_Abl_new/Abl1_{afMut}_copy/"
        copy_file_to_dir(file_A, file_B)

# get_af_ligand_pdbqt()

def get_af_ligand_pdb():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element2'].tolist()
    for mut in mutationList:
        afKey = mut
        afMut = afKey.split('_')[0]
        modi = afKey.split('_')[1]
        ri = afKey.split('_')[2]
        seedi = afKey.split('_')[3]
        afKpi = afKey.split('_')[4]
        file_A = glob.glob('/home/data1/BGM/mutationEffect/ensemble/autodock_Abl_new/Abl1_{}_*/Abl1_{}_*_model_{}_ptm_{}_{}_{}_*_vina_out_temp.pdb'.format(afMut, afMut, modi, ri, seedi, afKpi))[0]
        file_B = f"../data/af/autodock_Abl_new/Abl1_{afMut}_copy/"
        copy_file_to_dir(file_A, file_B)

# get_af_ligand_pdb()

def get_af_protein_pdbqt():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element2'].tolist()
    for mut in mutationList:
        afKey = mut
        afMut = afKey.split('_')[0]
        modi = afKey.split('_')[1]
        ri = afKey.split('_')[2]
        seedi = afKey.split('_')[3]
        afKpi = afKey.split('_')[4]
        file_A = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{}_*256_512_32/pdb/model_{}_ptm_{}_{}_{}_pocket.pdbqt'.format(afMut, modi, ri, seedi, afKpi))[0]
        file_B = f"../data/af/data_raw/unzip/Abl1_{afMut}_copy_256_512_32/pdb/"
        copy_file_to_dir(file_A, file_B)

# get_af_protein_pdbqt()

def get_af_protein_pdb():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element2'].tolist()
    for mut in mutationList:
        afKey = mut
        afMut = afKey.split('_')[0]
        modi = afKey.split('_')[1]
        ri = afKey.split('_')[2]
        seedi = afKey.split('_')[3]
        afKpi = afKey.split('_')[4]
        file_A = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{}_*256_512_32/pdb/model_{}_ptm_{}_{}_{}_pocket.pdb'.format(afMut, modi, ri, seedi, afKpi))[0]
        file_B = f"../data/af/data_raw/unzip/Abl1_{afMut}_copy_256_512_32/pdb/"
        copy_file_to_dir(file_A, file_B)

# get_af_protein_pdb()

def get_exp_contact():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element1'].tolist()
    for mut in mutationList:
        file_A = glob.glob('/home/data1/BGM/mutationEffect/data/pocket/contact/contact_{}*.csv'.format(mut.split('_')[0]))[0]
        file_B = f"../data/exp/pocket/contact/"
        copy_file_to_dir(file_A, file_B)

# get_exp_contact()

def get_exp_pdb():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element1'].tolist()
    for mut in mutationList:
        file_A = glob.glob('/home/data1/BGM/mutationEffect/ABL_AlphaFold/{}_*/{}_*.pdb'.format(mut.split('_')[0], mut.split('_')[0]))[0]
        file_B = f"../data/exp/ABL_AlphaFold/{mut.split('_')[0]}_copy/"
        copy_file_to_dir(file_A, file_B)

# get_exp_pdb()

def get_af_contact():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element2'].tolist()
    for mut in mutationList:
        afKey = mut
        afMut = afKey.split('_')[0]
        modi = afKey.split('_')[1]
        ri = afKey.split('_')[2]
        seedi = afKey.split('_')[3]
        afKpi = afKey.split('_')[4]
        mapAfName = f'{afKey}_contactMap.csv'
        file_A = '/home/data1/BGM/mdrugEffect/bigData/contact/{}'.format(mapAfName)
        file_B = f"../data/af/contact/"
        copy_file_to_dir(file_A, file_B)

# get_af_contact()

def get_af_pdb():
    mutationList = pd.read_csv('../data/resistance_top2_twin_muts_tuple_noGauss.csv')['element2'].tolist()
    for mut in mutationList:
        afKey = mut
        afMut = afKey.split('_')[0]
        modi = afKey.split('_')[1]
        ri = afKey.split('_')[2]
        seedi = afKey.split('_')[3]
        afKpi = afKey.split('_')[4]
        mapAfName = f'{afKey}_contactMap.csv'
        file_A = glob.glob('/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{}_filtering_*/pdb/model_{}_ptm_{}_{}.pdb'.format(afMut, modi, ri, seedi))[0]
        file_B = f"../data/af/data_raw/unzip/Abl1_{afMut}_filtering_copy/pdb/"
        copy_file_to_dir(file_A, file_B)

# get_af_pdb()

    