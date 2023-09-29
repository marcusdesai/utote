use crate::{Counter, Multiset};
use core::any::type_name;
use core::fmt::Formatter;
use core::marker::PhantomData;
use serde::{
    de::{self, Deserialize, Deserializer, SeqAccess, Visitor},
    ser::{Serialize, SerializeTuple, Serializer},
};

impl<T: Counter, const SIZE: usize> Serialize for Multiset<T, SIZE>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_tuple(SIZE)?;
        for elem in self {
            s.serialize_element(elem)?;
        }
        s.end()
    }
}

struct ArrayVisitor<T: Counter, const SIZE: usize> {
    _phantom: PhantomData<T>,
}

impl<'de, T: Counter, const SIZE: usize> Visitor<'de> for ArrayVisitor<T, SIZE>
where
    T: Deserialize<'de>,
{
    type Value = [T; SIZE];

    fn expecting(&self, formatter: &mut Formatter) -> core::fmt::Result {
        write!(formatter, "an array of {}, size {}", type_name::<T>(), SIZE)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut arr = [T::ZERO; SIZE];
        let mut idx = 0;
        while idx < SIZE {
            let opt_val = seq.next_element()?;
            let val = opt_val.ok_or_else(|| de::Error::invalid_length(idx + 1, &self))?;
            arr[idx] = val;
            idx += 1
        }
        Ok(arr)
    }
}

impl<'de, T: Counter, const SIZE: usize> Deserialize<'de> for Multiset<T, SIZE>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_tuple(
                SIZE,
                ArrayVisitor {
                    _phantom: PhantomData,
                },
            )
            .map(|arr| Multiset::from_array(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ser() {
        let ms = Multiset::from([5u8, 4, 3, 2, 1]);
        let serialized = serde_json::to_string(&ms).unwrap();
        assert_eq!("[5,4,3,2,1]", serialized);
    }

    #[test]
    fn test_de() {
        let serialized = "[5,4,3,2,1]";
        let deserialized: Multiset<u8, 5> = serde_json::from_str(serialized).unwrap();
        let ms = Multiset::from([5u8, 4, 3, 2, 1]);
        assert_eq!(ms, deserialized)
    }
}
