AP="$(autopep8 -r --diff --aggressive --aggressive .)"
echo "$AP"
test -z "$AP"
