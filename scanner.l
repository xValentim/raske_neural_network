%{
#include "parser.tab.h" // Inclui o arquivo gerado pelo Bison para sincronizar tokens
%}

digit       [0-9]
number      {digit}+
identifier  [a-zA-Z_][a-zA-Z0-9_]*
whitespace  [ \t\r\n]+

%%

"neural_network"                                { return NEURAL_NETWORK; }
"{"                                             { return OPEN_BRACE; }
"}"                                             { return CLOSE_BRACE; }
"print"                                         { return PRINT; }
"while"                                         { return WHILE; }
"if"                                            { return IF; }
"else"                                          { return ELSE; }
"add_input_layer"                               { return ADD_INPUT_LAYER; }
"add_dense_layer"                               { return ADD_DENSE_LAYER; }
"add_conv_layer"                                { return ADD_CONV_LAYER; }
"add_maxpooling_layer"                          { return ADD_MAXPOOLING_LAYER; }
"add_batch_normalization_layer"                 { return ADD_BATCH_NORMALIZATION_LAYER; }
"add_dropout_layer"                             { return ADD_DROPOUT_LAYER; }
"relu"|"sigmoid"|"softmax"|"tanh"|"linear"      { return ACTIVATION; }
"="                                             { return ASSIGN; }
"("                                             { return OPEN_PAREN; }
")"                                             { return CLOSE_PAREN; }
","                                             { return COMMA; }
"=="|">"|"<"                                    { return REL_OP; }
"+"|"-"                                         { return ADD_OP; }
"*"|"/"                                         { return MUL_OP; }
{number}                                        { yylval.num = atoi(yytext); return NUMBER; }
{identifier}                                    { yylval.str = strdup(yytext); return IDENTIFIER; }
{whitespace}                                    {/* ignorar espaços em branco */}

%%

int yywrap(void) {
  return 1;
}