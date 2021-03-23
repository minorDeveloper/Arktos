![Arktos_Banner@0.5x](./media/Arktos_Banner@0.5x.jpg)

Arktos
&middot;
[![Build Status](https://img.shields.io/github/workflow/status/minorDeveloper/Arktos/Arktos-Build)](https://github.com/minorDeveloper/Arktos/actions/workflows/build.yml)
[![Coveralls](https://img.shields.io/coveralls/github/minorDeveloper/Arktos)](https://coveralls.io/github/minorDeveloper/Arktos)
=====

Arktos is an Orbital Manoever Engine for the simulation and optimization of spacecraft trajectories in an n-body system. The project is primarily written in C++ with CUDA kernels available for increased performance.

This project utilizes the [Magnum](https://magnum.graphics/) engine for visualization using OpenGL.

To download the executable head on over to [releases](https://github.com/minorDeveloper/Arktos/releases) and grab the latest version!

## Features
 * It builds!

## Contents
-- [Features](#features)
-- [Building](#building)
-- [Usage and examples](#usage-and-examples)
-- [Development and contributing](#development-and-contributing)
-- [Acknowledgments](#acknowledgments)


## Building

Arktos is built using CMake, and its dependencies are included as git submodules (aside from SDL which is bundled for Windows builds).

Start by downloading the project.

```
git clone --recursive git://github.com/minorDeveloper/Arktos.git
cd Arktos
```


### Installing dependencies

#### Linux

```
sudo apt-get upgrade
sudo apt install libsdl2-dev
```

#### MacOS

```
brew install sdl2
```

#### Windows

SDL for windows is bundled with Arktos, no additional steps necessary.

### Build project

```
mkdir build && cd build
cmake build ..
cmake --build --config Debug
```

### Run tests

```
ctest -C Debug --verbose
```

## Usage and examples

TODO

## Development and contributing
I welcome pull requests and will take a look at any issues raised. If you want to see where the development of Arktos is headed then take a look at the [projects](https://github.com/minorDeveloper/Arktos/projects) page.

## Acknowledgments


![Arktos_Small@0.5x](./media/Arktos_Small@0.5x.jpg)
