# Boccia-bci-data-analysis
General repo for data analysis associated with the [boccia-bci](https://github.com/kirtonBCIlab/boccia-bci) repo.

The repo is organized in the following folders:
- [Data](./Data/): Data folders with naming `YYMMDD experiment description`.
- [Functions](./Functions/): General use functions for processing data.
- [Scripts](./Scripts/): Scripts to process the data. If a script is only applicable to a certain data folder,
the name of the script should match the date of the day in the folder

## Notes
- The files in the [Data](./Data/) folder should be uploated to the BCI4Kids Google Drive in `BCI studies/Boccia/2502 Boccia BCI Pilot data`.

## Miniconda environment

To set up the `conda` environment in this repo, install `miniconda` and run the following commands from terminal:

```bash
conda env create -f environment.yml
conda activate boccia
```

If new packages are added to the local environment, upload the modified version by using the following command from terminal:

```bash 
conda activate boccia
conda env export --no-builds > environment.yml
```
