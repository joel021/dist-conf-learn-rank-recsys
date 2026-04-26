import json
from recsysconfident.ml.models.multivaeracmodel import get_multivae_m_dl
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.datasets.movie_lens_reader import MovieLensReader


ml1m_info = DatasetInfo(**json.load(open('../data/ml-1m/info.json', 'r')),
                     database_name='ml-1m',
                     split_run_uri="../runs/", root_uri="../")

ml1m_df = MovieLensReader(ml1m_info).read()

ml1m_info.build(ml1m_df, None, True)

model, fit_dataloader, eval_dataloader, test_dataloader = get_multivae_m_dl(ml1m_info)

for batch in eval_dataloader:
    users, items, labels = batch

    mean, var = model.predict(users, items)
    print(f"{mean.shape} {var.shape}")
    break

