FROM osrf/ros:jazzy-desktop-full

# Example of installing programs
RUN apt-get update \
    && apt-get install -y \
    fd-find \
    git \
    npm \
    python3-pip \
    python3-poetry \
    ripgrep \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    && echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /root/.bashrc \
    && eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)" \
    && brew install neovim \
    && brew install zellij

# RUN pip install mujoco --break-system-packages

RUN git clone https://github.com/google-deepmind/mujoco_menagerie.git /root/mujoco_menagerie

