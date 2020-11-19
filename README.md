# FriendBlend

## Team Information
| Name | Roll Number| Branch |
| --- | --- | --- |
| Ahish Deshpande | 2018102022 | ECE |
| Pranav Kirsur | 2018101070 | CSE |
| Pranav Tadimeti | 2018101055 | CSE |
| Yoogottam Khandelwal | 2018101019 | CSE |

## Project code layout
```
.
├── app              # mobile app (abandoned after mid-eval)
├── documents        # presentation related files
├── images           # contains images we used for running the algorithm
├── Makefile          # helper commands for quickly building and running the project
├── misc             # random experiments we did while implementing the paper
├── README.md        # you are here
└── src              # the project implementation
```

The source code has been ordered in a logical manner instead of just putting everything into a single file.
```
src
├── Dockerfile
├── friendblend                      # main module
│   ├── global_vars.py               # contains module level per-process global variables
│   ├── helpers.py                   # basic helper functions (logging, timing, etc.)
│   ├── main.py                      # main code, the algorithm
│   ├── processing                   # contains code related to individual steps in the pipeline
│   │   ├── alpha_blending.py
│   │   ├── color_correction.py
│   │   ├── face_body_detection.py
│   │   ├── grab_cut.py
│   │   ├── helpers.py
│   │   ├── __init__.py
│   │   ├── keypoint.py
└── requirements.txt                 # pip install this
```

## How to run?
We assume that you will be running this on a linux system which has the GNU standard C Library (most of the normal machines have this).

### Prerequisites
 - `python3.8`
 - `pip` (corresponding to python3.8)  
OR  
 - `docker`

### The normal way
Since you most probably don't want to mess up your system's python installation, it's best if you create a python virtual environment.

```sh
project-appy-fizz$ pip install -U virtualenv
project-appy-fizz$ virtualenv venv -p python3.8
project-appy-fizz$ source venv/bin/activate
```

Install the requirements
```sh
project-appy-fizz$ pip install -r src/requirements.txt
```

Now, go inside the `src` directory.
```sh
project-appy-fizz$ cd src
```

Here, source the `.env` file which sets the environment variable `$PYTHONPATH`
```sh
project-appy-fizz/src$ source .env
```

Now, run `main.py` **from inside the same directory**
```sh
project-appy-fizz/src$ python friendblend/main.py
```
This runs it with the default fallback images. If you want to run this on your own images, please put them inside the `images` directory and

```sh
project-appy-fizz/src$ python friendblend/main.py F1.ext F2.ext
```
where files `F1.ext` and `F2.ext` are present in the `images` directory inside our repository.

**NOTE** `F1.ext` is just an example, naming `myfile.png` also works.

### The docker way
If you have docker installed, it's very easy to run our code.

Ensure you are at repository root.

To build the docker image,
```sh
project-appy-fizz$ make build
```

To run the docker image,
```sh
project-appy-fizz$ make run
```

Note that in order to run it with custom images, you'll have to run the actual `docker run` command instead of using the `Makefile`.

```sh
project-appy-fizz$ docker run --rm -it -v ${PWD}/images:/images friendblend F1.ext F2.ext
```

where files `F1.ext` and `F2.ext` images are present in the `images` directory inside our repository.

The outputs of each intermediate step of the friendblend algorithm, along with the final output, are stored in `images/outputs/` directory.
