echo "Downloading HumanML3D pretrained models"
mkdir -p save
cd save
gdown 1ZrJMJD4tLQZUURxDEWMl_DzFfGyVCFrn

unzip official_model.zip
echo -e "Cleaning\n"
rm official_model.zip

echo "Downloading done!"