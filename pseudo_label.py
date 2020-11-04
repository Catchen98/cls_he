import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--csv_file', default='submission.csv')
    arg('--save_file', default='pseudo_label.csv')
    arg('--prob_thr', default=0.99,type=float)

    args = parser.parse_args()
    sub_df = pd.read_csv(args.csv_file)
    pseudo_df = sub_df[sub_df.apply(lambda x:max(x['healthy'],x['rust'],x['scab'],x['multiple_diseases']), axis=1) >= args.prob_thr]
    pseudo_df['healthy'] = pseudo_df['healthy'].apply(lambda x:int(x>=args.prob_thr))
    pseudo_df['multiple_diseases'] = pseudo_df['multiple_diseases'].apply(lambda x:int(x>=args.prob_thr))
    pseudo_df['rust'] = pseudo_df['rust'].apply(lambda x:int(x>=args.prob_thr))
    pseudo_df['scab'] = pseudo_df['scab'].apply(lambda x:int(x>=args.prob_thr))

    assert pseudo_df.apply(lambda x: max(x['healthy'],x['rust'],x['scab'],x['multiple_diseases']), axis=1).sum() == pseudo_df.shape[0]
    print(pseudo_df.shape)
    pseudo_df.to_csv(args.save_file,index=False)
    # import pdb;pdb.set_trace()
