use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Length,
    Angle,
    Scalar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExprError(pub String);

impl std::fmt::Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(char),
}

fn lex(s: &str) -> Result<Vec<Tok>, ExprError> {
    if s.len() > 65_536 {
        return Err(ExprError("expression is too long".into()));
    }
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() || ch == '.' {
            let st = i;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.') {
                i += 1;
            }
            if i < c.len()
                && (c[i] == 'e' || c[i] == 'E')
                && i + 1 < c.len()
                && (c[i + 1].is_ascii_digit() || c[i + 1] == '-')
            {
                i += 2;
                while i < c.len() && c[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let t: String = c[st..i].iter().collect();
            out.push(Tok::Num(
                t.parse()
                    .map_err(|_| ExprError(format!("bad number '{t}'")))?,
            ));
        } else if ch.is_alphabetic() || ch == '_' {
            let st = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(c[st..i].iter().collect()));
        } else if "+-*/^()".contains(ch) {
            out.push(Tok::Op(ch));
            i += 1;
        } else if ch == '°' {
            out.push(Tok::Ident("deg".into()));
            i += 1;
        } else {
            return Err(ExprError(format!("unexpected '{ch}'")));
        }
    }
    Ok(out)
}

fn unit_factor(name: &str) -> Option<(f64, Unit)> {
    Some(match name {
        "mm" => (1.0, Unit::Length),
        "cm" => (10.0, Unit::Length),
        "m" => (1000.0, Unit::Length),
        "in" => (25.4, Unit::Length),
        "ft" => (304.8, Unit::Length),
        "deg" => (std::f64::consts::PI / 180.0, Unit::Angle),
        "rad" => (1.0, Unit::Angle),
        _ => return None,
    })
}

/// Values of named parameters (already in base units).
pub trait Scope {
    fn lookup(&self, name: &str) -> Option<f64>;
    fn dimensions(&self, _name: &str) -> [i32; 2] {
        [1, 0]
    }
}

impl Scope for HashMap<String, f64> {
    fn lookup(&self, name: &str) -> Option<f64> {
        self.get(name).copied()
    }
}

/// A value in base units with length/angle exponents. Bare arithmetic stays
/// scalar until addition to a quantity or the final field default is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantity {
    pub value: f64,
    dimensions: [i32; 2],
}
impl Quantity {
    fn scalar(value: f64) -> Self {
        Self {
            value,
            dimensions: [0, 0],
        }
    }
    fn unit(value: f64, unit: Unit) -> Self {
        Self {
            value,
            dimensions: match unit {
                Unit::Length => [1, 0],
                Unit::Angle => [0, 1],
                Unit::Scalar => [0, 0],
            },
        }
    }
    fn default_unit(self, unit: Unit) -> Self {
        Self::unit(
            if unit == Unit::Angle {
                self.value.to_radians()
            } else {
                self.value
            },
            unit,
        )
    }
    fn add(mut self, mut rhs: Self, subtract: bool) -> Result<Self, ExprError> {
        if self.dimensions != rhs.dimensions {
            let promote = |value: Self, dimensions: [i32; 2]| -> Result<Self, ExprError> {
                match dimensions {
                    [1, 0] => Ok(value.default_unit(Unit::Length)),
                    [0, 1] => Ok(value.default_unit(Unit::Angle)),
                    _ => Err(ExprError("incompatible quantities in addition".into())),
                }
            };
            if self.dimensions == [0, 0] {
                self = promote(self, rhs.dimensions)?;
            } else if rhs.dimensions == [0, 0] {
                rhs = promote(rhs, self.dimensions)?;
            } else {
                return Err(ExprError("cannot add lengths and angles".into()));
            }
        }
        self.value += if subtract { -rhs.value } else { rhs.value };
        Ok(self)
    }
    fn multiply(self, rhs: Self, divide: bool) -> Result<Self, ExprError> {
        if divide && rhs.value == 0. {
            return Err(ExprError("division by zero".into()));
        }
        let mut dimensions = [0; 2];
        for (i, d) in dimensions.iter_mut().enumerate() {
            *d = if divide {
                self.dimensions[i].checked_sub(rhs.dimensions[i])
            } else {
                self.dimensions[i].checked_add(rhs.dimensions[i])
            }
            .ok_or_else(|| ExprError("unit exponent overflow".into()))?;
        }
        Ok(Self {
            value: if divide {
                self.value / rhs.value
            } else {
                self.value * rhs.value
            },
            dimensions,
        })
    }
    fn power(self, rhs: Self) -> Result<Self, ExprError> {
        if rhs.dimensions != [0, 0] {
            return Err(ExprError("exponent must be scalar".into()));
        }
        let mut dimensions = [0; 2];
        for (i, d) in dimensions.iter_mut().enumerate() {
            let exponent = self.dimensions[i] as f64 * rhs.value;
            if !exponent.is_finite()
                || exponent.fract().abs() > 1e-10
                || exponent.abs() > i32::MAX as f64
            {
                return Err(ExprError("invalid dimensional exponent".into()));
            }
            *d = exponent as i32;
        }
        Ok(Self {
            value: self.value.powf(rhs.value),
            dimensions,
        })
    }
}

/// Typed parameter environment; the numeric compatibility API remains available.
#[derive(Clone, Debug, Default)]
pub struct Parameters(pub HashMap<String, Quantity>);
impl Scope for Parameters {
    fn lookup(&self, name: &str) -> Option<f64> {
        self.0.get(name).map(|q| q.value)
    }
    fn dimensions(&self, name: &str) -> [i32; 2] {
        self.0.get(name).map(|q| q.dimensions).unwrap_or([0, 0])
    }
}

struct Parser<'a> {
    t: Vec<Tok>,
    i: usize,
    scope: &'a dyn Scope,
    depth: usize,
}

impl Parser<'_> {
    fn nested(
        &mut self,
        parse: fn(&mut Self) -> Result<Quantity, ExprError>,
    ) -> Result<Quantity, ExprError> {
        if self.depth >= 64 {
            return Err(ExprError("expression nesting exceeds 64 levels".into()));
        }
        self.depth += 1;
        let result = parse(self);
        self.depth -= 1;
        result
    }
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn eat(&mut self, op: char) -> bool {
        if self.peek() == Some(&Tok::Op(op)) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Result<Quantity, ExprError> {
        let mut v = self.term()?;
        loop {
            if self.eat('+') {
                v = v.add(self.term()?, false)?;
            } else if self.eat('-') {
                v = v.add(self.term()?, true)?;
            } else {
                return Ok(v);
            }
        }
    }
    fn term(&mut self) -> Result<Quantity, ExprError> {
        let mut v = self.power()?;
        loop {
            if self.eat('*') {
                v = v.multiply(self.power()?, false)?;
            } else if self.eat('/') {
                v = v.multiply(self.power()?, true)?;
            } else {
                return Ok(v);
            }
        }
    }
    fn power(&mut self) -> Result<Quantity, ExprError> {
        let b = self.unary()?;
        if self.eat('^') {
            let exponent = self.nested(Self::power)?;
            b.power(exponent)
        } else {
            Ok(b)
        }
    }
    fn unary(&mut self) -> Result<Quantity, ExprError> {
        if self.eat('-') {
            {
                let mut q = self.nested(Self::unary)?;
                q.value = -q.value;
                Ok(q)
            }
        } else if self.eat('+') {
            self.nested(Self::unary)
        } else {
            self.atom()
        }
    }
    fn atom(&mut self) -> Result<Quantity, ExprError> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.i += 1;
                // Optional unit suffix; default unit otherwise.
                if let Some(Tok::Ident(u)) = self.peek().cloned() {
                    if let Some((f, unit)) = unit_factor(&u) {
                        self.i += 1;
                        return Ok(Quantity::unit(n * f, unit));
                    }
                }
                Ok(Quantity::scalar(n))
            }
            Some(Tok::Ident(name)) => {
                self.i += 1;
                if self.eat('(') {
                    let a = self.nested(Self::expr)?;
                    if !self.eat(')') {
                        return Err(ExprError("missing ')'".into()));
                    }
                    return match name.as_str() {
                        "sqrt" => a.power(Quantity::scalar(0.5)),
                        "abs" => Ok(Quantity {
                            value: a.value.abs(),
                            ..a
                        }),
                        "sin" | "cos" | "tan" => {
                            if a.dimensions != [0, 0] && a.dimensions != [0, 1] {
                                return Err(ExprError(
                                    "trigonometric arguments must be angles or scalar radians"
                                        .into(),
                                ));
                            }
                            Ok(Quantity::scalar(match name.as_str() {
                                "sin" => a.value.sin(),
                                "cos" => a.value.cos(),
                                _ => a.value.tan(),
                            }))
                        }
                        _ => Err(ExprError(format!("unknown function '{name}'"))),
                    };
                }
                if name == "pi" || name == "PI" {
                    return Ok(Quantity::scalar(std::f64::consts::PI));
                }
                self.scope
                    .lookup(&name)
                    .map(|value| Quantity {
                        value,
                        dimensions: self.scope.dimensions(&name),
                    })
                    .ok_or_else(|| ExprError(format!("unknown parameter '{name}'")))
            }
            Some(Tok::Op('(')) => {
                self.i += 1;
                let v = self.nested(Self::expr)?;
                if !self.eat(')') {
                    return Err(ExprError("missing ')'".into()));
                }
                Ok(v)
            }
            Some(t) => Err(ExprError(format!("unexpected {t:?}"))),
            None => Err(ExprError("empty expression".into())),
        }
    }
}

pub fn eval_quantity(src: &str, default: Unit, scope: &dyn Scope) -> Result<Quantity, ExprError> {
    let mut p = Parser {
        t: lex(src)?,
        i: 0,
        scope,
        depth: 0,
    };
    let mut q = p.expr()?;
    if p.i != p.t.len() {
        return Err(ExprError(format!("unexpected trailing input in '{src}'")));
    }
    if !q.value.is_finite() {
        return Err(ExprError("result is not finite".into()));
    }
    if q.dimensions == [0, 0] {
        q = q.default_unit(default);
    }
    Ok(q)
}

pub fn eval(src: &str, unit: Unit, scope: &dyn Scope) -> Result<f64, ExprError> {
    let q = eval_quantity(src, unit, scope)?;
    if q.dimensions != Quantity::unit(0., unit).dimensions {
        return Err(ExprError(format!("expression units do not match {unit:?}")));
    }
    Ok(q.value)
}

/// Identifiers referenced by an expression (for dependency ordering).
pub fn references(src: &str) -> Vec<String> {
    lex(src)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|t| match t {
            Tok::Ident(n)
                if unit_factor(&n).is_none()
                    && !["sqrt", "sin", "cos", "tan", "abs", "pi", "PI"].contains(&n.as_str()) =>
            {
                Some(n)
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn units_and_params() {
        let mut s = HashMap::new();
        s.insert("w".to_string(), 40.0);
        assert_eq!(eval("w / 2 + 1 cm", Unit::Length, &s).unwrap(), 30.0);
        assert!((eval("90", Unit::Angle, &s).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((eval("1 in", Unit::Length, &s).unwrap() - 25.4).abs() < 1e-12);
        assert_eq!(eval("-(2 + 3) * 2 ^ 2", Unit::Scalar, &s).unwrap(), -20.0);
        assert!(eval("nope * 2", Unit::Length, &s).is_err());
        assert_eq!(references("w * depth + 3 mm"), vec!["w", "depth"]);
    }

    #[test]
    fn scalar_arithmetic_and_explicit_units_are_distinct() {
        let s = HashMap::<String, f64>::new();
        for (expression, degrees) in [
            ("90 / 2", 45.),
            ("2 * 30 deg", 60.),
            ("30 deg * 2", 60.),
            ("30 deg + 2", 32.),
            ("1 rad + 90", 90. + 1f64.to_degrees()),
        ] {
            assert!(
                (eval(expression, Unit::Angle, &s).unwrap().to_degrees() - degrees).abs() < 1e-10,
                "{expression}"
            );
        }
        assert!((eval("sin(30 deg)", Unit::Scalar, &s).unwrap() - 0.5).abs() < 1e-12);
        assert!((eval("sin(pi / 2)", Unit::Scalar, &s).unwrap() - 1.).abs() < 1e-12);
        assert_eq!(eval("sqrt((4 mm)^2)", Unit::Length, &s).unwrap(), 4.);
        for expression in ["1 mm + 2 deg", "1 mm", "sqrt(4 deg)"] {
            assert!(eval(expression, Unit::Angle, &s).is_err());
        }
        assert!(eval("sin(1 mm)", Unit::Scalar, &s).is_err());
    }

    #[test]
    fn malformed_deep_expressions_return_errors_instead_of_overflowing_the_stack() {
        let s = HashMap::<String, f64>::new();
        for expression in [
            format!("{}1{}", "(".repeat(10000), ")".repeat(10000)),
            format!("{}1", "-".repeat(10000)),
            "2^".repeat(10000) + "1",
        ] {
            assert!(eval(&expression, Unit::Scalar, &s).is_err());
        }
    }
}
