%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int yylex();
extern int yyparse();
extern FILE* yyin;

void yyerror(const char* s);
%}

%union {
    int num;
    char* activation;
    char* identifier;
}

%token NEURAL_NETWORK
%token OPEN_BRACE CLOSE_BRACE
%token ASSIGN PRINT WHILE IF
%token ADD_INPUT_LAYER ADD_DENSE_LAYER ADD_CONV_LAYER
%token ADD_MAXPOOLING_LAYER ADD_BATCH_NORMALIZATION_LAYER ADD_DROPOUT_LAYER
%token ACTIVATION
%token NUMBER
%token <identifier> IDENTIFIER_TOKEN  /* Usando <identifier> para tokens de identificadores */
%type <num> NUMBER
%type <activation> ACTIVATION
%token ASSIGNMENT /* Definindo o token ASSIGNMENT */
%token ELSE /* Definindo o token ELSE */

%%

network: NEURAL_NETWORK OPEN_BRACE block CLOSE_BRACE { printf("Parsed a neural network\n"); }
       ;

block: statement { printf("Parsed a statement\n"); }
     | block statement { printf("Parsed a statement\n"); }
     ;

statement: ASSIGN '\n' { printf("Parsed an assignment statement\n"); } /* Usando ASSIGN em vez de ASSIGNMENT */
         | PRINT '(' EXPRESSION ')' '\n' { printf("Parsed a print statement\n"); }
         | WHILE '(' BOOL_EXP ')' OPEN_BRACE '\n' block CLOSE_BRACE { printf("Parsed a while loop\n"); }
         | IF '(' BOOL_EXP ')' OPEN_BRACE '\n' block CLOSE_BRACE { printf("Parsed an if statement\n"); }
         | IF '(' BOOL_EXP ')' OPEN_BRACE '\n' block CLOSE_BRACE ELSE OPEN_BRACE '\n' block CLOSE_BRACE { printf("Parsed an if-else statement\n"); }
         | ADD_INPUT_LAYER '(' NUMBER ',' NUMBER ')' '\n' { printf("Parsed an add input layer statement\n"); }
         | ADD_DENSE_LAYER '(' NUMBER ',' ACTIVATION ')' '\n' { printf("Parsed an add dense layer statement\n"); }
         | ADD_CONV_LAYER '(' NUMBER ',' NUMBER ',' NUMBER ',' ACTIVATION ')' '\n' { printf("Parsed an add conv layer statement\n"); }
         | ADD_MAXPOOLING_LAYER '(' NUMBER ')' '\n' { printf("Parsed an add maxpooling layer statement\n"); }
         | ADD_BATCH_NORMALIZATION_LAYER '(' ')' '\n' { printf("Parsed an add batch normalization layer statement\n"); }
         | ADD_DROPOUT_LAYER '(' NUMBER ')' '\n' { printf("Parsed an add dropout layer statement\n"); }
         ;


BOOL_EXP: BOOL_TERM { printf("Parsed a boolean expression\n"); }
        | BOOL_EXP "or" BOOL_TERM { printf("Parsed a boolean expression\n"); }
        ;

BOOL_TERM: REL_EXP { printf("Parsed a boolean term\n"); }
         | BOOL_TERM "and" REL_EXP { printf("Parsed a boolean term\n"); }
         ;

REL_EXP: EXPRESSION { printf("Parsed a relational expression\n"); }
       | EXPRESSION "==" EXPRESSION { printf("Parsed a relational expression\n"); }
       | EXPRESSION ">" EXPRESSION { printf("Parsed a relational expression\n"); }
       | EXPRESSION "<" EXPRESSION { printf("Parsed a relational expression\n"); }
       ;

EXPRESSION: TERM { printf("Parsed an expression\n"); }
          | EXPRESSION '+' TERM { printf("Parsed an expression\n"); }
          | EXPRESSION '-' TERM { printf("Parsed an expression\n"); }
          ;

TERM: FACTOR { printf("Parsed a term\n"); }
     | TERM '*' FACTOR { printf("Parsed a term\n"); }
     | TERM '/' FACTOR { printf("Parsed a term\n"); }
     ;

FACTOR: NUMBER { printf("Parsed a factor\n"); }
      | IDENTIFIER_TOKEN { printf("Parsed an identifier\n"); }  /* Usando IDENTIFIER_TOKEN */
      | '+' FACTOR { printf("Parsed a factor\n"); }
      | '-' FACTOR { printf("Parsed a factor\n"); }
      | "not" FACTOR { printf("Parsed a factor\n"); }
      | '(' EXPRESSION ')' { printf("Parsed a factor\n"); }
      | "read" '(' ')' { printf("Parsed a factor\n"); }
      ;


%%

void yyerror(const char* s) {
    printf("Parser error: %s\n", s);
    exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    yyin = fopen(argv[1], "r");
    if (!yyin) {
        printf("Failed to open input file\n");
        return 1;
    }
    yyparse();
    fclose(yyin);
    return 0;
}
