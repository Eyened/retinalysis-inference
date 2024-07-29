from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rtnls_inference",
    version="0.1.0",
    author="Eyened Team",
    author_email="j.vargasquiros@erasmusmc.nl",
    description="Retinal model inference package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": []},
    install_requires=[
        "numpy == 1.*",
        "pandas == 2.*",
        "opencv-python == 4.*",
        "joblib==1.*",
        "Pillow == 9.*",
        "click==8.*",
        "pydicom==2.3.1",
        "nbstripout==0.3.9",
        "lightning==2.1.0",
        "albumentations==1.3.1",
        "PyYAML>=5.4",
        "monai==1.2.0",
    ],
    python_requires=">=3.10",
)
