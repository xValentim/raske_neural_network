%{
#include <stdio.h>
#include <stdlib.h>
void yyerror(const char *s);
%}

%union {
    int num;
    char *str;
}

%token <str> IDENTIFIER NUMBER ACTIVATION
%token NEURAL_NETWORK OPEN_BRACE CLOSE_BRACE PRINT WHILE IF ELSE
%token ADD_INPUT_LAYER ADD_DENSE_LAYER ADD_CONV_LAYER ADD_MAXPOOLING_LAYER ADD_BATCH_NORMALIZATION_LAYER ADD_DROPOUT_LAYER
%token ASSIGN OPEN_PAREN CLOSE_PAREN COMMA REL_OP ADD_OP MUL_OP

%type <str> expression

%%
neural_network: NEURAL_NETWORK OPEN_BRACE block CLOSE_BRACE
                {
                    printf("Parsed a neural network definition.\n");
                };

block: 
    | block statement
    ;

statement:
    assignment
  | print
  | while
  | if
  | add_layer
  ;

assignment: IDENTIFIER ASSIGN expression;

print: PRINT OPEN_PAREN expression CLOSE_PAREN;

while: WHILE OPEN_PAREN expression CLOSE_PAREN OPEN_BRACE block CLOSE_BRACE;

if: IF OPEN_PAREN expression CLOSE_PAREN OPEN_BRACE block CLOSE_BRACE
    | IF OPEN_PAREN expression CLOSE_PAREN OPEN_BRACE block CLOSE_BRACE ELSE OPEN_BRACE block CLOSE_BRACE;

add_layer:
    ADD_INPUT_LAYER OPEN_PAREN expression CLOSE_PAREN { printf("Added input layer.\n");}
  | ADD_INPUT_LAYER OPEN_PAREN expression COMMA expression CLOSE_PAREN { printf("Added input layer.\n");}
  | ADD_DENSE_LAYER OPEN_PAREN expression COMMA ACTIVATION CLOSE_PAREN { printf("Added dense layer.\n");}
  | ADD_CONV_LAYER OPEN_PAREN expression COMMA expression COMMA expression COMMA ACTIVATION CLOSE_PAREN { printf("Added convolutional layer.\n");}
  | ADD_MAXPOOLING_LAYER OPEN_PAREN expression CLOSE_PAREN { printf("Added maxpooling layer.\n");}
  | ADD_BATCH_NORMALIZATION_LAYER OPEN_PAREN CLOSE_PAREN { printf("Added batch normalization layer.\n");}
  | ADD_DROPOUT_LAYER OPEN_PAREN expression CLOSE_PAREN { printf("Added dropout layer.\n");}
  ;

expression:
    NUMBER 
  | IDENTIFIER 
  | expression ADD_OP expression 
  | expression MUL_OP expression 
  | expression REL_OP expression 
  ;

%%
int main(void) {
    yyparse();
    return 0;
}

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}
