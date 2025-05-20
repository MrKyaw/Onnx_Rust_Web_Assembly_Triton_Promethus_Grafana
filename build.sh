
#!/bin/bash

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null
then
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build the WASM package
wasm-pack build --target web --release

# The output will be in pkg/ directory