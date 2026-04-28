# SOCcer
Standardized Occupational Coding for Computer-assisted Epidemiologic Research

soccer-rs is our attempt to move our code to rust.  This way the code to crosswalk, 
preprocess, and run SOCcerNET and CLIPS is the same in R, Python, or in browsers (WASM).

All of the required data is pre-baked into the application, but the first time you run you will have
to download the embedding model from huggingface and our classifier from github.  They are fairly small
but too large to stick in the application.