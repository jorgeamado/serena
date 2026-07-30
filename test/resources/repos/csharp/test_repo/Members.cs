using System;

namespace TestProject
{
    public class Bag
    {
        public int Value { get; set; }
        public int Count;
        public event Action? Changed;

        public void Fire() => Changed?.Invoke();
    }

    public class BagUser
    {
        public int Read(Bag b) => b.Value;
        public void Write(Bag b) => b.Value = 5;
        public void Bump(Bag b) => b.Value += 1;
        public int ReadField(Bag b) => b.Count;
        public void WriteField(Bag b) => b.Count = 3;
        public void Sub(Bag b, Action h) => b.Changed += h;
        public void Unsub(Bag b, Action h) => b.Changed -= h;
    }
}
