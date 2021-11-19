[Image Classification System]

-Objective: 
> [1] This project is intended to provide a full scale environment where image data generation and classifying model generation functionalities are available through tensorflow and/or pytorch API. \
> [2] Throughout the development of this project, I have had tons of issues trying to implement necessary tools from scratch. This opened new learning opportunities in which I had the chance to analyze required components of the proeject, break them down into sub-problems, and merge their solutions as a whole. \
> [3] Although tensorflow and pytorch backends provide such utilities, I have decided to build them one by one with my own logics. 


-Image Data: 
> [1] Description : Rectangle, triangle, and circle images are generated through customized data augmentation technique ( randomized variations in width, height, location, etc. ) \
> [2] Size: ( these are default settings, please configure according to your needs. ) \
> --> Training set   ( 1,000 images ) \
> --> Validation set ( 300 images ) \
> --> Test set       ( 200 images ) \


-Usage: 
> $ python run.py tf      ( for tensorflow backend) \
> $ python run.py torch   ( for pytorch backend) \


-Output:
> The prediction results will be printed on your terminal. 


-Note: 
> [1] This codebase may be used for benchmarking purposes, such as model quality comparison between the two deep-learning frameworks or customization of image data ( you may need to implement such methods that satisfy your own needs ). \
> [2] Existing image data and model files guarantee that the corresponding file generation steps are skipped for direct access to test dataset prediction.
