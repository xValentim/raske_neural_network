import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

class Token:
    def __init__(self, type: str, value: int):
        self.type = type
        self.value = value
        
class SymbolTable:
    def __init__(self):
        self.symbols = {}
    
    def get(self, name):
        value, type = self.symbols.get(name)
        if value is None:
            raise ValueError("Variável não definida: " + name)
        return (value, type)
    
    def create(self, name):
        self.symbols[name] = (None, None)
    
    def set(self, name, value):
        if name not in self.symbols:
            raise ValueError("Variável não definida: " + name)
        
        typeof = value[1]
        valueof = value[0]
        self.symbols[name] = (valueof, typeof)
    
    def set_model(self, name, value):
        if name not in self.symbols:
            raise ValueError("Variável não definida: " + name)
        
        self.symbols[name] = (value, 'MODEL')

class FuncTable:
    def __init__(self):
        self.funcs = {}
    
    def get(self, key):
        value = self.funcs.get(key)
        if value is None:
            raise ValueError("Função não definida: " + key)
        return value
    
    def set(self, key, value):
        if key in self.funcs:
            raise ValueError("Função já definida: " + key)
        self.funcs[key] = value
        
        
class Node:
    def __init__(self, value, children):
        self.value = value
        self.children = children
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        if self.value == 'INT':
            return self.children[0]
        elif self.value == 'PLUS':
            return self.children[0].evaluate(symbol_table, func_table) + self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'MINUS':
            return self.children[0].evaluate(symbol_table, func_table) - self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'TIMES':
            return self.children[0].evaluate(symbol_table, func_table) * self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'DIVIDE':
            return self.children[0].evaluate(symbol_table, func_table) // self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'OR':
            return self.children[0].evaluate(symbol_table, func_table) or self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'AND':
            return self.children[0].evaluate(symbol_table, func_table) and self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'NOT':
            return not self.children[0].evaluate(symbol_table, func_table)
        elif self.value == 'LESS':
            return self.children[0].evaluate(symbol_table, func_table) < self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'GREATER':
            return self.children[0].evaluate(symbol_table, func_table) > self.children[1].evaluate(symbol_table, func_table)
        elif self.value == 'EQUAL':
            return self.children[0].evaluate(symbol_table, func_table) == self.children[1].evaluate(symbol_table, func_table)
        else:
            raise ValueError("Operação inválida")
 
class NeuralNetwork(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        model = keras.Sequential()
        symbol_table.create('base_model')
        symbol_table.set_model('base_model', model)
        for child in self.children:
            child.evaluate(symbol_table, func_table)
        return symbol_table.get('base_model')[0]

class Identifier(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        return symbol_table.get(self.value.value)
    
class Assign(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        var = self.children[0]
        if var.value not in symbol_table.symbols:
            raise ValueError("Variável não definida: " + var.value)
        result = self.children[1].evaluate(symbol_table, func_table)
        symbol_table.set(var.value, result)
        return result
 
class Print(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        print(self.children[0].evaluate(symbol_table, func_table)[0])
        
class While(Node):
    def __init__(self, value, children):
        super().__init__(value, children)

    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        while self.children[0].evaluate(symbol_table, func_table)[0]:
            self.children[1].evaluate(symbol_table, func_table)

class If(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        condition = self.children[0].evaluate(symbol_table, func_table)
        result = condition[0]
        
        if result == 1:
            self.children[1].evaluate(symbol_table, func_table)
        else:
            self.children[2].evaluate(symbol_table, func_table)

class Return(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        return self.children[0].evaluate(symbol_table, func_table)

class Block(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        for child in self.children:
            if child.value.type == 'RETURN':
                return child.evaluate(symbol_table, func_table)
            child.evaluate(symbol_table, func_table)
            
class AddFlattenLayer(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        model = symbol_table.get('base_model')[0]
        model.add(keras.layers.Flatten())
        print('add_flatten_layer')
            
class AddInputLayer(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        model = symbol_table.get('base_model')[0]
        if len(self.children) == 1:
            child_1 = self.children[0].evaluate(symbol_table, func_table)
            input_shape = [child_1[0]]
            model.add(keras.layers.Input(shape=input_shape))            
            print('add_input_layer with input shape: ', input_shape)

        else:
            child_1 = self.children[0].evaluate(symbol_table, func_table)
            child_2 = self.children[1].evaluate(symbol_table, func_table)
            input_shape = [child_1[0], child_2[0]]
            model.add(keras.layers.Input(shape=input_shape))
            model.add(keras.layers.Reshape((child_1[0], child_2[0], 1)))
            print('add_input_layer with input shape: ', input_shape)

        
        
class AddDenseLayer(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        model = symbol_table.get('base_model')[0]
        neurons = self.children[0].evaluate(symbol_table, func_table)[0]
        activation = self.children[1].value
        model.add(keras.layers.Dense(neurons, activation=activation))
        print(f'add_dense_layer with {neurons} neurons and activation {activation}')

class AddConvLayer(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        model = symbol_table.get('base_model')[0]
        filters = self.children[0].evaluate(symbol_table, func_table)[0]
        kernel_size = self.children[1].evaluate(symbol_table, func_table)[0]
        stride = self.children[2].evaluate(symbol_table, func_table)[0]
        activation = self.children[3].value
        print('add_conv_layer with filters:', filters, 'kernel_size:', kernel_size, 'stride:', stride, 'activation:', activation)
        model.add(keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(stride, stride), activation=activation))
        
class AddMaxPoolingLayer(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        pool_size = self.children[0].evaluate(symbol_table, func_table)[0]
        model = symbol_table.get('base_model')[0]
        model.add(keras.layers.MaxPooling2D(pool_size))
        print('add_maxpooling_layer with pool_size:', pool_size)
        
class AddBatchNormalizationLayer(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        model = symbol_table.get('base_model')[0]
        model.add(keras.layers.BatchNormalization())
        print('add_batch_normalization_layer')
        
class AddDropoutLayer(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        model = symbol_table.get('base_model')[0]
        perc_dropout = self.children[0].evaluate(symbol_table, func_table)[0] / 100
        model.add(keras.layers.Dropout(perc_dropout))
        print('add_dropout_layer with perc_dropout:', perc_dropout)
        


class FuncCall(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        func_name = self.value.value
        func = func_table.get(func_name)
        if len(func.children) - 2 != len(self.children):
            raise ValueError("Número de argumentos inválido")
        
        new_symbol_table = SymbolTable()
        for i in range(len(self.children)):
            new_symbol_table.create(func.children[i + 1].value.value)
            new_symbol_table.set(func.children[i + 1].value.value, self.children[i].evaluate(symbol_table, func_table))
        
        
        result = func.children[-1].evaluate(new_symbol_table, func_table)
        return result
    
class FuncDec(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        func_name = self.children[0].value.value
        func_table.set(func_name, self)
        
class BinOp(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        
        if self.value.type in ['PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'OR', 'AND'] and (self.children[0].evaluate(symbol_table, func_table)[1] != 'INT' or self.children[1].evaluate(symbol_table, func_table)[1] != 'INT'):
            raise ValueError("Operação inválida")
        
        if self.value.type in ['LESS', 'GREATER', 'EQUAL'] and (self.children[0].evaluate(symbol_table, func_table)[1] != self.children[1].evaluate(symbol_table, func_table)[1]):
            raise ValueError("Operação inválida")
        
        
        if self.value.type == 'PLUS':
            return self.children[0].evaluate(symbol_table, func_table)[0] + self.children[1].evaluate(symbol_table, func_table)[0], 'INT'
        
        elif self.value.type == 'MINUS':
            return self.children[0].evaluate(symbol_table, func_table)[0] - self.children[1].evaluate(symbol_table, func_table)[0], 'INT'
        
        elif self.value.type == 'TIMES':
            return self.children[0].evaluate(symbol_table, func_table)[0] * self.children[1].evaluate(symbol_table, func_table)[0], 'INT'
        
        elif self.value.type == 'DIVIDE':
            return self.children[0].evaluate(symbol_table, func_table)[0] // self.children[1].evaluate(symbol_table, func_table)[0], 'INT'
        
        elif self.value.type == 'OR':
            return (self.children[0].evaluate(symbol_table, func_table)[0] or self.children[1].evaluate(symbol_table, func_table)[0]) * 1, 'INT'
        
        elif self.value.type == 'AND':
            return (self.children[0].evaluate(symbol_table, func_table)[0] and self.children[1].evaluate(symbol_table, func_table)[0]) * 1, 'INT'
        
        elif self.value.type == 'LESS':
            if self.children[0].evaluate(symbol_table, func_table)[1] == 'STRING':
                return (self.children[0].evaluate(symbol_table, func_table) < self.children[1].evaluate(symbol_table, func_table)) * 1, 'STRING'
            return (self.children[0].evaluate(symbol_table, func_table) < self.children[1].evaluate(symbol_table, func_table)) * 1, 'INT'
        
        elif self.value.type == 'GREATER':
            if self.children[0].evaluate(symbol_table, func_table)[1] == 'STRING':
                return (self.children[0].evaluate(symbol_table, func_table) > self.children[1].evaluate(symbol_table, func_table)) * 1, 'STRING'
            return (self.children[0].evaluate(symbol_table, func_table) > self.children[1].evaluate(symbol_table, func_table)) * 1, 'INT'
        
        elif self.value.type == 'EQUAL':
            if self.children[0].evaluate(symbol_table, func_table)[1] == 'STRING':
                return (self.children[0].evaluate(symbol_table, func_table) == self.children[1].evaluate(symbol_table, func_table)) * 1, 'STRING'
            return (self.children[0].evaluate(symbol_table, func_table) == self.children[1].evaluate(symbol_table, func_table)) * 1, 'INT'
        
        elif self.value.type == 'CONCAT':
            return str(self.children[0].evaluate(symbol_table, func_table)[0]) + str(self.children[1].evaluate(symbol_table, func_table)[0]), 'STRING'
        
        else:
            raise ValueError("Operação inválida")

class UnOp(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        if self.children[0].evaluate(symbol_table, func_table)[1] != 'INT':
            raise ValueError("Operação inválida")
        
        if self.value.type == 'MINUS':
            return -self.children[0].evaluate(symbol_table, func_table)[0], 'INT'
        elif self.value.type == 'PLUS':
            return self.children[0].evaluate(symbol_table, func_table)[0], 'INT'
        elif self.value.type == 'NOT':
            return not self.children[0].evaluate(symbol_table, func_table)[0], 'INT'
        else:
            raise ValueError("Operação inválida")

class IntVal(Node):
    def __init__(self, value):
        super().__init__(Token('INT', value), [])
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        return (self.value.value, 'INT')

class StringVal(Node):
    def __init__(self, value):
        super().__init__(Token('STRING', value), [])
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        return (self.value.value, 'STRING')

class VarDec(Node):
    def __init__(self, value, children):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        if len(self.children) == 1:
            var = self.children[0]
            if var.value in symbol_table.symbols:
                raise ValueError("Variável já definida: " + var.value)
            symbol_table.create(var.value)
            return (None, None)
        
        var = self.children[0]
        if var.value in symbol_table.symbols:
            raise ValueError("Variável já definida: " + var.value)
        symbol_table.create(var.value)
        symbol_table.set(var.value, self.children[1].evaluate(symbol_table, func_table))
        return (None, None)
        
    
class NoOp(Node):
    def __init__(self, value=None, children=None):
        super().__init__(value, children)
    
    def evaluate(self, symbol_table: SymbolTable, func_table: FuncTable):
        return (None, None)

class PrePro:
    def __init__(self, code):
        self.code = self.filter(code)
        
    # Remove comments from lua file
    def filter(self, code):
        code = code.split('\n')
        new_code = []
        for line in code:
            if '--' in line:
                line = line.split('--')[0]
            new_code.append(line)
        return '\n'.join(new_code)
        


class Tokenizer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.actual = None
        self.aux_token = None

    def select_next(self):
        operators = {'+' : 'PLUS',
                     '-' : 'MINUS',
                     '*' : 'TIMES',
                     '/' : 'DIVIDE',
                     '(' : 'LPAREN',
                     ')' : 'RPAREN',
                     '=' : 'EQUALS',
                     'print' : 'PRINT',
                     
                     'and' : 'AND',
                     'or' : 'OR',
                     'not' : 'NOT',
                     '==' : 'EQUAL',
                     '<' : 'LESS',
                     '>' : 'GREATER',
                     '<=' : 'LESS_EQUAL',
                     '>=' : 'GREATER_EQUAL',
                     'while': 'WHILE',
                     'if' : 'IF',
                     'do' : 'DO',
                     'then' : 'THEN',
                     'else' : 'ELSE',
                     'end' : 'END',
                     'read' : 'READ',
                     ',': 'COMMA',
                     'return': 'RETURN',
                     'function': 'FUNCTION',
                     '..' : 'CONCAT',
                     'string' : 'STRING',
                     'local' : 'LOCAL',
                     '\n' : 'EOL',
                     
                     # New tokens
                     '{' : 'LBRACE',
                     '}' : 'RBRACE',
                     
                     'add_input_layer': 'ADD_INPUT_LAYER',
                     'add_dense_layer': 'ADD_DENSE_LAYER',
                     'add_conv_layer': 'ADD_CONV_LAYER',
                     'add_maxpooling_layer': 'ADD_MAXPOOLING_LAYER',
                     'add_batch_normalization_layer': 'ADD_BATCH_NORMALIZATION_LAYER',
                     'add_dropout_layer': 'ADD_DROPOUT_LAYER',
                     'add_flatten_layer': 'ADD_FLATTEN_LAYER',
                     'neural_network': 'NEURAL_NETWORK',
                     
                     'relu': 'RELU',
                     'sigmoid': 'SIGMOID',
                     'softmax': 'SOFTMAX',
                     'tanh': 'TANH',
                     'linear': 'LINEAR',
                     
                     }
        
        while self.position < len(self.source) and (self.source[self.position] == ' ' or self.source[self.position] == '\t'):
            self.position += 1
            
        if self.position >= len(self.source):
            return Token('EOF', None)
        
        elif self.source[self.position].isdigit():
            num = 0
            while self.position < len(self.source) and self.source[self.position].isdigit():
                num = num * 10 + int(self.source[self.position])
                self.position += 1

            self.aux_token = Token('INT', int(num))
        
        elif self.source[self.position] == ',':
            self.aux_token = Token('COMMA', ',')
            self.position += 1
            
        elif self.source[self.position] == '"':
            string = ''
            self.position += 1
            while self.position < len(self.source) and self.source[self.position] != '"':
                string += self.source[self.position]
                self.position += 1
            if self.position >= len(self.source):
                raise ValueError("String não fechada")
            self.position += 1
            self.aux_token = Token('STRING', string)
            
        # Add Variable
        elif self.source[self.position].isalpha() or self.source[self.position] == '_':
            var = ''
            while self.position < len(self.source) and (self.source[self.position].isalpha() or self.source[self.position].isdigit() or self.source[self.position] == '_'):
                var += self.source[self.position]
                self.position += 1
                
            if var == 'print':
                self.aux_token = Token('PRINT', var)
            elif var == 'while':
                self.aux_token = Token('WHILE', var)
            elif var == 'if':
                self.aux_token = Token('IF', var)
            elif var == 'do':
                self.aux_token = Token('DO', var)
            elif var == 'then':
                self.aux_token = Token('THEN', var)
            elif var == 'else':
                self.aux_token = Token('ELSE', var)
            elif var == 'end':
                self.aux_token = Token('END', var)
            elif var == 'and':
                self.aux_token = Token('AND', var)
            elif var == 'or':
                self.aux_token = Token('OR', var)
            elif var == 'not':
                self.aux_token = Token('NOT', var)
            elif var == 'read':
                self.aux_token = Token('READ', var)
            elif var == 'local':
                self.aux_token = Token('LOCAL', var)
            elif var == 'return':
                self.aux_token = Token('RETURN', var)
            elif var == 'function':
                self.aux_token = Token('FUNCTION', var)
            elif var == 'add_input_layer':
                self.aux_token = Token('ADD_INPUT_LAYER', var)
            elif var == 'add_dense_layer':
                self.aux_token = Token('ADD_DENSE_LAYER', var)
            elif var == 'add_conv_layer':
                self.aux_token = Token('ADD_CONV_LAYER', var)
            elif var == 'add_maxpooling_layer':
                self.aux_token = Token('ADD_MAXPOOLING_LAYER', var)
            elif var == 'add_batch_normalization_layer':
                self.aux_token = Token('ADD_BATCH_NORMALIZATION_LAYER', var)
            elif var == 'add_dropout_layer':
                self.aux_token = Token('ADD_DROPOUT_LAYER', var)
            elif var == 'add_flatten_layer':
                self.aux_token = Token('ADD_FLATTEN_LAYER', var)
            elif var == 'relu':
                self.aux_token = Token('RELU', var)
            elif var == 'sigmoid':
                self.aux_token = Token('SIGMOID', var)
            elif var == 'softmax':
                self.aux_token = Token('SOFTMAX', var)
            elif var == 'tanh':
                self.aux_token = Token('TANH', var)
            elif var == 'linear':
                self.aux_token = Token('LINEAR', var)
            elif var == 'neural_network':
                self.aux_token = Token('NEURAL_NETWORK', var)
            else:
                self.aux_token = Token('IDENTIFIER', var)
            
        elif self.source[self.position] == '=' and self.source[self.position + 1] == '=':
            self.aux_token = Token('EQUAL', '==')
            self.position += 2
            
        elif self.source[self.position] == '.' and self.source[self.position + 1] == '.':
            self.aux_token = Token('CONCAT', '..')
            self.position += 2
        
        elif self.source[self.position] == '=':
            self.aux_token = Token('EQUALS', '=')
            self.position += 1        
        
        elif self.source[self.position] == '<':
            self.aux_token = Token('LESS', '<')
            self.position += 1
        
        elif self.source[self.position] == '>':
            self.aux_token = Token('GREATER', '>')
            self.position += 1
        
        elif self.source[self.position] == '\n':
            self.aux_token = Token('EOL', '\n')
            self.position += 1
        
        elif self.source[self.position] in operators:
            self.aux_token = Token(operators[self.source[self.position]], self.source[self.position])
            self.position += 1
            
        else:
            raise ValueError("Caracter inválido: " + self.source[self.position])
        
        return self.aux_token


class Parser:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.actual_token = self.tokenizer.select_next()

    def parse_factor(self):
        aux_operator = {'MINUS' : -1, 'PLUS' : 1}
        op = self.actual_token.type
        padding_symbol_table = SymbolTable()
        if self.actual_token.type == 'INT':
            result = self.actual_token.value
            result = IntVal(result)
            self.actual_token = self.tokenizer.select_next()
            return result
        
        elif self.actual_token.type in ['MINUS', 'PLUS', 'NOT']:
            self.actual_token = self.tokenizer.select_next()
            if op == 'MINUS':
                result = UnOp(Token('MINUS', '-'), [self.parse_factor()])
            elif op == 'PLUS':
                result = UnOp(Token('PLUS', '+'), [self.parse_factor()])
            elif op == 'NOT':
                result = UnOp(Token('NOT', 'not'), [self.parse_factor()])
            return result
        
        elif self.actual_token.type == 'IDENTIFIER':
            result = Identifier(self.actual_token, [])
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type == 'LPAREN':
                self.actual_token = self.tokenizer.select_next()
                
                if self.actual_token.type == 'RPAREN':
                    self.actual_token = self.tokenizer.select_next()
                    return FuncCall(Token('IDENTIFIER', result.value.value), [])
                
                elif self.actual_token.type == 'IDENTIFIER' or self.actual_token.type == 'INT' or self.actual_token.type == 'STRING':
                    
                    
                    result_block = []
                    while self.actual_token.type != 'RPAREN':
                        
                        if self.actual_token.type != 'IDENTIFIER' and self.actual_token.type != 'INT' and self.actual_token.type != 'STRING':
                            raise ValueError("Esperado identificador após '('")
                        
                        result_block.append(self.parse_bool_expression())
                        
                        if self.actual_token.type != 'COMMA' and self.actual_token.type != 'RPAREN':
                            raise ValueError("Esperado ',' ou ')' após argumento")
                        
                        if self.actual_token.type == 'COMMA':
                            self.actual_token = self.tokenizer.select_next()
                    
                    if self.actual_token.type != 'RPAREN':
                        raise ValueError("Esperado ')' após argumento")
                    
                    self.actual_token = self.tokenizer.select_next()
                    
                    
                    return FuncCall(Token('IDENTIFIER', result.value.value), result_block)
            
            return result
        
        
        
        elif self.actual_token.type == 'READ':
            
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após read")
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após read")
            self.actual_token = self.tokenizer.select_next()
            
            return IntVal(int(input()))
        
        elif self.actual_token.type == 'STRING':
            result = StringVal(self.actual_token.value)
            self.actual_token = self.tokenizer.select_next()
            return result
        
        elif self.actual_token.type == 'LPAREN':
            self.actual_token = self.tokenizer.select_next()
            result = self.parse_bool_expression()
            
            
            
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após expressão")
            self.actual_token = self.tokenizer.select_next()
            return result
        else:
            raise ValueError("Token inesperado: " + self.actual_token.type)
    

    def parse_term(self):
        result = self.parse_factor()
        while self.actual_token.type in ['TIMES', 'DIVIDE']:
            if self.actual_token.type == 'TIMES':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('TIMES', '*'), [result, self.parse_factor()])
                
                
            elif self.actual_token.type == 'DIVIDE':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('DIVIDE', '/'), [result, self.parse_factor()])
                
        return result

    def parse_expression(self):
        result = self.parse_term()
        while self.actual_token.type in ['PLUS', 'MINUS', 'CONCAT']:
            if self.actual_token.type == 'PLUS':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('PLUS', '+'), [result, self.parse_term()])
            elif self.actual_token.type == 'MINUS':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('MINUS', '-'), [result, self.parse_term()])
            elif self.actual_token.type == 'CONCAT':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('CONCAT', '..'), [result, self.parse_term()])
            
        return result
    
    def parse_block(self):
        if self.actual_token.type != 'NEURAL_NETWORK':
            raise ValueError("Esperado neural_network")
        self.actual_token = self.tokenizer.select_next()
        
        if self.actual_token.type != 'LBRACE':
            raise ValueError("Esperado '{' após neural_network")
        self.actual_token = self.tokenizer.select_next()
        
        result = []
        while self.actual_token.type != 'EOF' and self.actual_token.type != 'RBRACE':
            result.append(self.parse_statement())
        return Block(Token('BLOCK', 'block'), result)
        
    def parse_statement(self):
        if self.actual_token.type == 'PRINT':
            self.actual_token = self.tokenizer.select_next()
            return Print(Token('PRINT', 'print'), [self.parse_bool_expression()])
        
        if self.actual_token.type == 'ADD_INPUT_LAYER':
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após add_input_layer")
            self.actual_token = self.tokenizer.select_next()
            input_value = self.parse_bool_expression()
            
            if self.actual_token.type != 'RPAREN' and self.actual_token.type != 'COMMA':
                raise ValueError("Esperado ')' ou ',' após add_input_layer")
            
            if self.actual_token.type == 'COMMA':
                self.actual_token = self.tokenizer.select_next()
                input_value_aux = self.parse_bool_expression()
                if self.actual_token.type != 'RPAREN':
                    raise ValueError("Esperado ')' após add_input_layer")
                self.actual_token = self.tokenizer.select_next()
                return AddInputLayer(Token('ADD_INPUT_LAYER', 'add_input_layer'), [input_value, input_value_aux])
                
            else:
                self.actual_token = self.tokenizer.select_next()
                return AddInputLayer(Token('ADD_INPUT_LAYER', 'add_input_layer'), [input_value])
                
        if self.actual_token.type == 'ADD_DENSE_LAYER':
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após add_input_layer")
            self.actual_token = self.tokenizer.select_next()
            neurons = self.parse_bool_expression()
            
            if self.actual_token.type != 'COMMA':
                raise ValueError("Esperado ',' após neurônios")
            self.actual_token = self.tokenizer.select_next()
            activation = self.actual_token
            
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após add_dense_layer")
            self.actual_token = self.tokenizer.select_next()
            
            
            return AddDenseLayer(Token('ADD_DENSE_LAYER', 'add_dense_layer'), [neurons, activation])
        
        if self.actual_token.type == 'ADD_CONV_LAYER':
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após add_input_layer")
            self.actual_token = self.tokenizer.select_next()
            filters = self.parse_bool_expression()
            
            if self.actual_token.type != 'COMMA':
                raise ValueError("Esperado ',' após filtros")
            self.actual_token = self.tokenizer.select_next()
            kernel_size = self.parse_bool_expression()
            
            if self.actual_token.type != 'COMMA':
                raise ValueError("Esperado ',' após kernel_size")
            self.actual_token = self.tokenizer.select_next()
            stride = self.parse_bool_expression()
            
            if self.actual_token.type != 'COMMA':
                raise ValueError("Esperado ',' após stride")
            self.actual_token = self.tokenizer.select_next()            
            activation = self.actual_token
            
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após add_dense_layer")
            self.actual_token = self.tokenizer.select_next()
            
            return AddConvLayer(Token('ADD_CONV_LAYER', 'add_conv_layer'), [filters, kernel_size, stride, activation])
        
        if self.actual_token.type == 'ADD_MAXPOOLING_LAYER':
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após add_maxpooling_layer")
            self.actual_token = self.tokenizer.select_next()
            pool_size = self.parse_bool_expression()
            
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após add_maxpooling_layer")
            self.actual_token = self.tokenizer.select_next()
            
            return AddMaxPoolingLayer(Token('ADD_MAXPOOLING_LAYER', 'add_maxpooling_layer'), [pool_size])
        
        if self.actual_token.type == 'ADD_BATCH_NORMALIZATION_LAYER':
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após add_batch_normalization_layer")
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após add_batch_normalization_layer")
            self.actual_token = self.tokenizer.select_next()
            
            return AddBatchNormalizationLayer(Token('ADD_BATCH_NORMALIZATION_LAYER', 'add_batch_normalization_layer'), [])
        
        if self.actual_token.type == 'ADD_FLATTEN_LAYER':
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após add_flatten_layer")
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após add_flatten_layer")
            self.actual_token = self.tokenizer.select_next()
            
            return AddFlattenLayer(Token('ADD_FLATTEN_LAYER', 'add_flatten_layer'), [])
        
        if self.actual_token.type == 'ADD_DROPOUT_LAYER':
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após add_dropout_layer")
            self.actual_token = self.tokenizer.select_next()
            perc = self.parse_bool_expression()
            
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após add_dropout_layer")
            self.actual_token = self.tokenizer.select_next()
            
            return AddDropoutLayer(Token('ADD_DROPOUT_LAYER', 'add_dropout_layer'), [perc])
        
        elif self.actual_token.type == 'LOCAL':
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'IDENTIFIER':
                raise ValueError("Esperado identificador após local")
            var = self.actual_token
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type == 'EQUALS':
                self.actual_token = self.tokenizer.select_next()
                return VarDec(Token('EQUALS', '='), [var, self.parse_bool_expression()])
            else:
                return VarDec(Token('LOCAL', 'local'), [var])
        
        elif self.actual_token.type == 'RETURN':
            self.actual_token = self.tokenizer.select_next()
            return Return(Token('RETURN', 'return'), [self.parse_bool_expression()])
        
        elif self.actual_token.type == 'FUNCTION':
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'IDENTIFIER':
                raise ValueError("Esperado identificador após function")
            func_name = Identifier(self.actual_token, [])
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após identificador")
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type == 'RPAREN':
                self.actual_token = self.tokenizer.select_next()
                if self.actual_token.type != 'EOL':
                    raise ValueError("Esperado 'EOL' após argumentos")
                self.actual_token = self.tokenizer.select_next()
                result_block = []
                while self.actual_token.type != 'END':
                    result_block.append(self.parse_statement())
                block = Block(Token('BLOCK', 'block'), result_block)
                self.actual_token = self.tokenizer.select_next()
                return FuncDec(Token('FUNCTION', 'function'), [func_name, block])
            
            
            var_decs = []
            while self.actual_token.type != 'RPAREN':
                var = self.actual_token
                
                if self.actual_token.type != 'IDENTIFIER':
                    raise ValueError("Esperado identificador após '('")
                
                child = Identifier(self.actual_token, [])
                
                var_decs.append(VarDec(Token('LOCAL', var.value), [child]))
                
                self.actual_token = self.tokenizer.select_next()
                
                if self.actual_token.type != 'COMMA' and self.actual_token.type != 'RPAREN':
                    raise ValueError("Esperado ',' ou ')' após argumento")
                
                if self.actual_token.type == 'COMMA':
                    self.actual_token = self.tokenizer.select_next()
                
            
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type != 'EOL':
                raise ValueError("Esperado 'EOL' após argumentos")
            
            self.actual_token = self.tokenizer.select_next()
            
            result_block = []
            while self.actual_token.type != 'END':
                result_block.append(self.parse_statement())
            block = Block(Token('BLOCK', 'block'), result_block)
            self.actual_token = self.tokenizer.select_next()
            return FuncDec(Token('FUNCTION', 'function'), [func_name] + var_decs + [block])
            
        elif self.actual_token.type == 'IDENTIFIER':
            var = self.actual_token
            self.actual_token = self.tokenizer.select_next()
            if self.actual_token.type == 'EQUALS':
                self.actual_token = self.tokenizer.select_next()
                
                return Assign(Token('EQUALS', '='), [var, self.parse_bool_expression()])
            
            elif self.actual_token.type == 'LPAREN':
                self.actual_token = self.tokenizer.select_next()
                
                if self.actual_token.type == 'RPAREN':
                    self.actual_token = self.tokenizer.select_next()
                    return FuncCall(Token('IDENTIFIER', var.value), [])
                
                elif self.actual_token.type == 'IDENTIFIER' or self.actual_token.type == 'INT' or self.actual_token.type == 'STRING':
                    
                    result_block = []
                    while self.actual_token.type != 'RPAREN':
                        
                        if self.actual_token.type != 'IDENTIFIER' and self.actual_token.type != 'INT' and self.actual_token.type != 'STRING':
                            raise ValueError("Esperado identificador após '('")
                        
                        result_block.append(self.parse_bool_expression())
                        
                        if self.actual_token.type != 'COMMA' and self.actual_token.type != 'RPAREN':
                            raise ValueError("Esperado ',' ou ')' após argumento")
                        
                        if self.actual_token.type == 'COMMA':
                            self.actual_token = self.tokenizer.select_next()
                    
                    if self.actual_token.type != 'RPAREN':
                        raise ValueError("Esperado ')' após argumento")
                    
                    self.actual_token = self.tokenizer.select_next()
                            
                    return FuncCall(Token('IDENTIFIER', var.value), result_block)
            
            else:
                raise ValueError("Token inesperado: " + self.actual_token.type)
        
        elif self.actual_token.type == 'WHILE':
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após while")
            self.actual_token = self.tokenizer.select_next()
            
            condition = self.parse_bool_expression()
            
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após condição")
            self.actual_token = self.tokenizer.select_next()
            
            
            if self.actual_token.type != 'LBRACE':
                raise ValueError("Esperado '{' após while")
            self.actual_token = self.tokenizer.select_next()
           
            if self.actual_token.type != 'EOL':
                raise ValueError("Esperado 'EOL' após do")
            self.actual_token = self.tokenizer.select_next()
            
            result_block = []
            while self.actual_token.type != 'RBRACE':
                result_block.append(self.parse_statement())
            block = Block(Token('BLOCK', 'block'), result_block)
            self.actual_token = self.tokenizer.select_next()
            
            return While(Token('WHILE', 'while'), [condition, block])
        
        elif self.actual_token.type == 'IF':
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'LPAREN':
                raise ValueError("Esperado '(' após if")
            
            self.actual_token = self.tokenizer.select_next()
    
            condition = self.parse_bool_expression()
    
            if self.actual_token.type != 'RPAREN':
                raise ValueError("Esperado ')' após condição")
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type != 'LBRACE':
                raise ValueError("Esperado '{' após if")
            self.actual_token = self.tokenizer.select_next()
           
            if self.actual_token.type != 'EOL':
                raise ValueError("Esperado 'EOL' após then")
            self.actual_token = self.tokenizer.select_next()
            
            result_block = []
            while self.actual_token.type != 'RBRACE':
                result_block.append(self.parse_statement())
            block = Block(Token('BLOCK', 'block'), result_block)
            self.actual_token = self.tokenizer.select_next()
            
            if self.actual_token.type == 'ELSE':
                self.actual_token = self.tokenizer.select_next()
                
                if self.actual_token.type != 'LBRACE':
                    raise ValueError("Esperado '{' após else")
                self.actual_token = self.tokenizer.select_next()
                
                if self.actual_token.type != 'EOL':
                    raise ValueError("Esperado 'EOL' após else")
                self.actual_token = self.tokenizer.select_next()
                
                result_block_else = []
                while self.actual_token.type != 'RBRACE':
                    result_block_else.append(self.parse_statement())
                block_else = Block(Token('BLOCK', 'block'), result_block_else)
                self.actual_token = self.tokenizer.select_next()
                
                return If(Token('IF', 'if'), [condition, block, block_else])
            else:
                if self.actual_token.type != 'EOL':
                    raise ValueError("Esperado 'EOL' após bloco")
                return If(Token('IF', 'if'), [condition, block, NoOp(Token('NOOP', 'noop'))])
        
        elif self.actual_token.type == 'EOL' :
            self.actual_token = self.tokenizer.select_next()
            return NoOp(Token('EOL', '\n'), [])

        else:
            raise ValueError("Token inesperado: " + self.actual_token.type)
        
    def parse_bool_expression(self):
        result = self.parse_bool_term()
        while self.actual_token.type in ['OR']:
            if self.actual_token.type == 'OR':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('OR', 'or'), [result, self.parse_bool_term()])
        return result
    
    def parse_bool_term(self):
        result = self.parse_rel_expression()
        while self.actual_token.type in ['AND']:
            if self.actual_token.type == 'AND':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('AND', 'and'), [result, self.parse_rel_expression()])
        return result
    
    def parse_rel_expression(self):
        result = self.parse_expression()
        while self.actual_token.type in ['EQUAL', 'LESS', 'GREATER']:
            if self.actual_token.type == 'EQUAL':
                # print('EQUAL')
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('EQUAL', '=='), [result, self.parse_expression()])
            elif self.actual_token.type == 'LESS':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('LESS', '<'), [result, self.parse_expression()])
            elif self.actual_token.type == 'GREATER':
                self.actual_token = self.tokenizer.select_next()
                result = BinOp(Token('GREATER', '>'), [result, self.parse_expression()])
        return result
        
    
    def run(self):
        symbol_table = SymbolTable()
        func_table = FuncTable()
        block = self.parse_block()
        neural_network = NeuralNetwork(Token('NEURAL_NETWORK', 'neural_network'), [block])
        return neural_network.evaluate(symbol_table, func_table)

# Open lua file
def open_file(path):
    try:
        with open(path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        sys.stderr.write("Erro: Arquivo não encontrado")
        sys.exit(1)


def main():
    path = sys.argv[1]
    code = open_file(path)
    code_filtered = PrePro(code).code
    arg = code_filtered
    if len(sys.argv) < 2:
        sys.stderr.write("Erro: Argumento inválido")
    elif len(arg) == 0 or arg[0] in ['+', '-', '*', '/'] or arg[-1] in ['+', '-', '*', '/'] or arg.replace(' ', '') == '':
        sys.stderr.write("Erro: Argumento inválido")
    else:
        tokenizer = Tokenizer(arg)
        parser = Parser(tokenizer)
        model = parser.run()
        
        model.save(f"{path.split('.')[0]}.h5")
        with open(f"{path.split('.')[0]}.summary",'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        keras.utils.plot_model(model, 
                               to_file=f"{path.split('.')[0]}.png", 
                               show_shapes=True, 
                               show_layer_names=True, 
                               show_layer_activations=True)

if __name__ == "__main__":
    main()