## Raske - Linguagem de Programação

### Descrição

O Raske é uma linguagem de programação desenvolvida especificamente para simplificar a criação e definição de arquiteturas de redes neurais de forma intuitiva e eficiente. Com o Raske, os desenvolvedores podem descrever a estrutura de suas redes neurais de maneira concisa e legível, sem a necessidade de lidar diretamente com a complexidade do código de implementação.

### Características

- **Sintaxe Simples**: A sintaxe da linguagem é intuitiva e fácil de aprender, permitindo que os desenvolvedores criem arquiteturas de redes neurais de forma rápida e eficaz.
- **Flexibilidade**: O Raske oferece flexibilidade para adicionar uma variedade de camadas, como camadas densas, de convolução, de dropout e outras, com diferentes configurações de parâmetros.
- **Suporte para Loops e Estruturas Condicionais**: A linguagem permite o uso de loops while e estruturas condicionais if, possibilitando a criação de arquiteturas de redes neurais dinâmicas e complexas.

## EBNF

```ebnf

BLOCK           ::= '{' STATEMENT '}';
STATEMENT       ::= ( "λ" | ASSIGNMENT | PRINT | WHILE | IF | NEURAL_NETWORK | ADD_LAYER ), "\n" ;
ASSIGNMENT      ::= IDENTIFIER '=' EXPRESSION ;
PRINT           ::= "print" '(' EXPRESSION ')' ;
WHILE           ::= "while" BOOL_EXP "do" "\n" "λ" '{' STATEMENT '}', "end" ;
IF              ::= "if" BOOL_EXP "then" "\n" "λ" '{' STATEMENT '}', ( "λ" | "else" "\n" "λ" '{' STATEMENT '}' ), "end" ;
NEURAL_NETWORK ::= "neural_network" '{' '}' ;
ADD_LAYER       ::= "add_layer" '(' LAYER_TYPE, LAYER_PARAMETERS ')' ';' ;
LAYER_TYPE      ::= "input_layer" | "hidden_layer" | "output_layer" | "convolutional_layer" | ... ;
LAYER_PARAMETERS ::= PARAMETER | PARAMETER ',' LAYER_PARAMETERS ;
PARAMETER       ::= PARAM_NAME '=' PARAM_VALUE ;
PARAM_NAME      ::= "neurons" | "activation_function" | "kernel_size" | ... ;
PARAM_VALUE     ::= NUMBER | STRING | ... ;
BOOL_EXP        ::= BOOL_TERM ('or' BOOL_TERM)* ;
BOOL_TERM       ::= REL_EXP ('and' REL_EXP)* ;
REL_EXP         ::= EXPRESSION (REL_OP EXPRESSION)* ;
EXPRESSIION     ::= TERM ('+' TERM | '-' TERM)* ;
TERM            ::= FACTOR ('*' FACTOR | '/' FACTOR)* ;
FACTOR          ::= NUMBER | IDENTIFIER | ('+' | '-' | 'not') FACTOR | '(' EXPRESSION ')' | "read" '(' ')' ;
IDENTIFIER      ::= LETTER (LETTER | DIGIT | '_')* ;
NUMBER          ::= DIGIT+ ;
LETTER          ::= ('a' ... 'z' | 'A' ... 'Z') ;
DIGIT           ::= ('0' ... '9') ;
REL_OP          ::= '==' | '>' | '<' ;

```

## Exemplo de uso

```C

neural_network {
    // Camada de entrada
    add_layer('input', 784);

    // Camadas ocultas
    int i = 0;
    while (i < 3) {
        if (i % 2 == 0) {
            add_layer('dense', 128, 'relu');
        } else {
            add_layer('dense', 64, 'relu');
        }
        add_layer('dropout', 0.25); // Dropout para regularização
        i = i + 1;
    }

    // Mais camadas ocultas
    i = 0;
    while (i < 2) {
        add_layer('dense', 32, 'relu');
        if (i == 1) {
            add_layer('dropout', 0.5); // Dropout mais agressivo para regularização
        }
        i = i + 1;
    }

    // Camada de saída
    add_layer('output', 10, 'softmax');
}




```