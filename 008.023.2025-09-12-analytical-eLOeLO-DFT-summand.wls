#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
assume=a\[Element]Integers&&b\[Element]Integers&&chimin\[Element]PositiveReals&&chimax\[Element]PositiveReals&&chi\[Element]PositiveReals&&chip\[Element]PositiveReals&&L\[Element]PositiveReals&&chimin<chimax&&chimin < chi && chimin < chip&&chi<chimax&&chip<chimax
L=chimax-chimin
Integrand[a_,b_]:=1/L^2 Cos[2Pi/L ((chib - chi)a)]Cos[ 2Pi/L((chib-chip)b)]/chib^2


(* ::Input:: *)
$Assumptions=assume;

argv = Rest @ $ScriptCommandLine;
argc = Length @ argv;
a0 = ToExpression[argv[[1]]]
b0 = ToExpression[argv[[2]]]

Print[ToString[a0]<>", "<>ToString[b0]];
tmp0=Integrate[Integrand[a0,b0], {chib, chimin, chimax},Assumptions->assume];
Export[StringJoin["008.023.mathematica/008.023.eLOeLO_summand_",ToString[a0],"_",ToString[b0],"_.txt"],tmp0]

Print["done"]
