import os
from prody import parsePDB, matchChains, measure
import pandas as pd
import glob
from multiprocessing import Pool

def process_single_df(df_name):
    print(df_name)
    mut = df_name.split('_')[0]
    modi = df_name.split('_')[1]
    ri = df_name.split('_')[2]
    seedi = df_name.split('_')[3]
    lig = df_name.split('_')[4]
    
    wt_af_pdb = f'/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_WILD_600k_filtering_8dc32_256_512_32/pdb/model_{modi}_ptm_{ri}_{seedi}.pdb'
    mt_af_pdb = glob.glob(f'/home/data1/BGM/mutationEffect/ensemble/data_raw/unzip/Abl1_{mut}*/pdb/model_{modi}_ptm_{ri}_{seedi}.pdb')[0]
    
    conf1 = parsePDB(wt_af_pdb, subset='ca')
    conf2 = parsePDB(mt_af_pdb, subset='ca')
    
    atommap1, atommap2, seqid, overlap = matchChains(conf1, conf2)[0]
    ca_coord2 = atommap2.getCoords()
    
    distances1 = []
    for i in range(len(ca_coord2)):
        distances_tmp = []
        for j in range(len(ca_coord2)):
            dist = measure.calcDistance(ca_coord2[i], ca_coord2[j])
            distances_tmp.append(dist / 10 if dist < 10 else 0)
        
        df0 = pd.DataFrame(distances_tmp)
        df1 = pd.DataFrame(distances1)
        df_merge = pd.concat([df1, df0], axis=1)
        distances1 = df_merge
    
    df = pd.DataFrame(distances1)
    output_path = f'/home/data1/BGM/mdrugEffect/bigData/contact/{df_name}_contactMap.csv'
    df.to_csv(output_path, index=False, header=False)

def main():
    df_list = pd.read_csv('/home/data1/BGM/mdrugEffect/bigData/resistance_All.csv')['key'].tolist()
    
    with Pool(6) as pool:
        pool.map(process_single_df, df_list)

if __name__ == '__main__':
    main()
