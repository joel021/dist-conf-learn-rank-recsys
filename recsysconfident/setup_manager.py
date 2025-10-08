import json

from torch import optim

from recsysconfident.environment import Environment
from recsysconfident.ml.fit.fit import train_model
from recsysconfident.setup import Setup


def export_history(setup, history, sufix):

    with open(f"{setup.instance_dir}/history-{setup.split_position}{sufix}.json", "w") as f:
        json.dump(history, f, indent=4)

def setup_fit(setup: Setup, model, fit_dl, val_dl, environ: Environment, device):

    if hasattr(model, 'train_method') and model.train_method is not None:
        history = model.train_method(model=model,
                                     fit_dl=fit_dl,
                                     val_dl=val_dl,
                                     environ=environ,
                                     epochs=50,
                                     device=device,
                                     patience=setup.patience)
        export_history(setup, history, "")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=setup.learning_rate
                               )
        history = train_model(model, training_loader=fit_dl, validation_loader=val_dl, environ=environ,
                              optimizer=optimizer, epochs=100, device=device, patience=setup.patience)
        export_history(setup, history, "standard")

        if hasattr(model, 'switch_to_second_train'):
            print(f"Model probs trained. Switching to ranking training.")
            model.switch_to_second_train()
            history = train_model(model, training_loader=fit_dl, validation_loader=val_dl, environ=environ,
                                  optimizer=optimizer, epochs=100, device=device, patience=setup.patience)
            model.switch_to_main_train()
            export_history(setup, history, "second-train")

    return model
