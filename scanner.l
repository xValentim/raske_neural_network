%{
#include "parser.tab.h"
%}

%%
"neural_network"                 { return NEURAL_NETWORK; }
"{"                              { return OPEN_BRACE; }
"}"                              { return CLOSE_BRACE; }
"="                              { return ASSIGN; }
"print"                          { return PRINT; }
"while"                          { return WHILE; }
"if"                             { return IF; }
"add_input_layer"                { return ADD_INPUT_LAYER; }
"add_dense_layer"                { return ADD_DENSE_LAYER; }
"add_conv_layer"                 { return ADD_CONV_LAYER; }
"add_maxpooling_layer"           { return ADD_MAXPOOLING_LAYER; }
"add_batch_normalization_layer"  { return ADD_BATCH_NORMALIZATION_LAYER; }
"add_dropout_layer"              { return ADD_DROPOUT_LAYER; }
"relu"                           { yylval.activation = strdup(yytext); return ACTIVATION; }
"sigmoid"                        { yylval.activation = strdup(yytext); return ACTIVATION; }
"softmax"                        { yylval.activation = strdup(yytext); return ACTIVATION; }
"tanh"                           { yylval.activation = strdup(yytext); return ACTIVATION; }
"linear"                         { yylval.activation = strdup(yytext); return ACTIVATION; }
[0-9]+                           { yylval.num = atoi(yytext); return NUMBER; }
[ \t\n]                          ; // Ignorar espaços em branco e tabulações
.                                { return yytext[0]; }
%%

int yywrap() {
    return 1;
}
