
#ifndef STD_EXPERIMENTAL_FUNDAMENTALS_V3_MDSPAN_HEADER
#define STD_EXPERIMENTAL_FUNDAMENTALS_V3_MDSPAN_HEADER

#define HOST_DEVICE /* __host__ __device__ */

#include <cassert>
#include <type_traits>
#include <utility>
#include <array>
#include <initializer_list>

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

enum : ptrdiff_t { dynamic_extent = -1 };

// [mdspan.extents]
template< ptrdiff_t ... StaticExtents >
class extents;

// [mdspan.extents.compare]
template<ptrdiff_t... LHS, ptrdiff_t... RHS>
HOST_DEVICE
constexpr bool operator==(const extents<LHS...>& lhs,
                          const extents<RHS...>& rhs) noexcept;

template<ptrdiff_t... LHS, ptrdiff_t... RHS>
HOST_DEVICE
constexpr bool operator!=(const extents<LHS...>& lhs,
                          const extents<RHS...>& rhs) noexcept;

// [mdspan.layout]
class layout_right ;
class layout_left ;
class layout_stride ;

// [mdspan.accessor.basic]
template<class ElementType>
class accessor_basic;

// [mdspan.basic]
template<class ElementType,
         class Extents,
         class LayoutPolicy = layout_right,
         class AccessorPolicy = accessor_basic<ElementType> >
class basic_mdspan ;

// [msspan.subspan]

namespace detail {
template<class ElementType,
         class Extents,
         class LayoutPolicy,
         class AccessorPolicy,
         class ... SliceSpecifiers>
class subspan_deduction;
}

template<class ElementType,
         class Extents,
         class LayoutPolicy,
         class AccessorPolicy,
         class ... SliceSpecifiers>
HOST_DEVICE
  typename detail::subspan_deduction<ElementType,
                                     Extents,
                                     LayoutPolicy,
                                     AccessorPolicy,
                                     SliceSpecifiers...>::type
subspan(const basic_mdspan<ElementType,Extents,LayoutPolicy,AccessorPolicy> &,
        SliceSpecifiers...) noexcept ;

class all_type { public: constexpr explicit all_type() = default; };

/* inline */ constexpr all_type all ;

}}} // std::experimental::fundamentals_v3

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#include <mdspan_helper.hpp>

namespace std {
namespace experimental {
namespace fundamentals_v3 {

template< ptrdiff_t ... StaticExtents >
class extents : private detail::extents_helper<0,StaticExtents...>
{
private:

  template< ptrdiff_t... > friend class extents ;

  using helper  = detail::extents_helper<0,StaticExtents...> ;

public:

  using index_type = ptrdiff_t ;

  HOST_DEVICE
  constexpr extents() noexcept : helper() {}

  HOST_DEVICE
  constexpr extents( extents && ) noexcept = default ;

  HOST_DEVICE
  constexpr extents( const extents & ) noexcept = default ;

  template< class ... IndexType >
  constexpr explicit extents( ptrdiff_t dn,
                              IndexType ... DynamicExtents ) noexcept
    : helper( dn , DynamicExtents... ) 
    { static_assert( 1+sizeof...(DynamicExtents) == helper::RankDynamic , "" ); }

  template<ptrdiff_t... OtherStaticExtents>
  extents( const extents<OtherStaticExtents...>& other )
    : helper( (const detail::extents_helper<0,OtherStaticExtents...> &) other ) {}

  HOST_DEVICE
  extents & operator = ( extents && ) noexcept = default;

  HOST_DEVICE
  extents & operator = ( const extents & ) noexcept = default;

  template<ptrdiff_t... OtherStaticExtents>
  extents & operator = ( const extents<OtherStaticExtents...>& other )
    { helper::operator=( (const detail::extents_helper<0,OtherStaticExtents...> &) other ); return *this ; }

  HOST_DEVICE
  ~extents() = default ;

  // [mdspan.extents.obs]

  HOST_DEVICE
  static constexpr size_t rank() noexcept
    { return sizeof...(StaticExtents); }

  HOST_DEVICE
  static constexpr size_t rank_dynamic() noexcept 
    { return helper::RankDynamic ; }

  HOST_DEVICE
  static constexpr index_type static_extent(size_t k) noexcept
    { return helper::static_extent(k); }

  HOST_DEVICE
  constexpr index_type extent(size_t k) const noexcept
    { return helper::extent(k); }

  // implementation details

    template<size_t K>
    HOST_DEVICE
    constexpr index_type extent() const noexcept
      { return helper::template extent<K>(); }

    HOST_DEVICE constexpr index_type extent() const noexcept
      { return helper::N ; }

    HOST_DEVICE constexpr const typename helper::next_t & next() const noexcept
      { return (const typename helper::next_t &) (*this); };

    HOST_DEVICE constexpr index_type product(size_t i, size_t j) const noexcept
      { return helper::product(i,j); }

    template<ptrdiff_t... OtherStaticExtents>
    HOST_DEVICE constexpr bool
    equal( const extents<OtherStaticExtents...>& other ) const noexcept
      { return rank() == other.rank() && helper::equal( other ); }
};

//----------------------------------------------------------------------------

template<ptrdiff_t... LHS, ptrdiff_t... RHS>
HOST_DEVICE
constexpr bool operator==(const extents<LHS...>& lhs,
                          const extents<RHS...>& rhs) noexcept
{ return lhs.equal(rhs); }

}}} // std::experimental::fundamentals_v3

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

struct layout_none {

  template<class Extents>
  class mapping {
  private:

    static_assert( Extents::rank() <= 1 , "" );

    Extents m_extents ;

  public:

    using index_type = ptrdiff_t ;

    HOST_DEVICE
    constexpr mapping() noexcept = default ;

    HOST_DEVICE
    constexpr mapping( mapping && ) noexcept = default ;

    HOST_DEVICE
    constexpr mapping( const mapping & ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( mapping && ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( const mapping & ) noexcept = default ;

    HOST_DEVICE
    constexpr mapping( const Extents & ext ) noexcept
      : m_extents( ext ) {}

    constexpr const Extents & extents() const noexcept { return m_extents ; }

    template<class... Indices>
    constexpr mapping( Indices... DynamicExtents ) noexcept
      : m_extents( DynamicExtents... ) {}

    constexpr index_type required_span_size() const noexcept
      { return m_extents.extent(0); }

    constexpr index_type operator()() const noexcept { return 0 ; }
    constexpr index_type operator()( index_type i ) const noexcept { return i ; }

    static constexpr bool is_always_unique     = true ;
    static constexpr bool is_always_contiguous = true ;
    static constexpr bool is_always_strided    = true ;

    constexpr bool is_unique()     const noexcept { return true ; }
    constexpr bool is_contiguous() const noexcept { return true ; }
    constexpr bool is_strided()    const noexcept { return true ; }

    static constexpr index_type stride(size_t) noexcept { return 1 ; }
  }; // struct mapping

}; // struct layout_none

}}} // std::experimental::fundamentals_v3

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

struct layout_right {

  template<class Extents>
  class mapping {
  private:

    Extents m_extents ;

  public:

    using index_type = ptrdiff_t ;

    HOST_DEVICE
    constexpr mapping() noexcept = default ;

    HOST_DEVICE
    constexpr mapping( mapping && ) noexcept = default ;

    HOST_DEVICE
    constexpr mapping( const mapping & ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( mapping && ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( const mapping & ) noexcept = default ;

    HOST_DEVICE
    constexpr mapping( const Extents & ext ) noexcept
      : m_extents( ext ) {}

    constexpr const Extents & extents() const noexcept { return m_extents ; }

    template<class... Indices>
    constexpr mapping( Indices... DynamicExtents ) noexcept
      : m_extents( DynamicExtents... ) {}

  private:

    // ( ( ( ( i0 ) * N1 + i1 ) * N2 + i2 ) * N3 + i3 ) ...

    template<class Ext, class ... Indices >
    static constexpr index_type
    offset( const Ext & ext,
            index_type sum,
            index_type i) noexcept
      { return sum * ext.extent() + i ; }

    template<class Ext, class ... Indices >
    static constexpr index_type
    offset( const Ext & ext,
            index_type sum,
            index_type i,
            Indices... indices ) noexcept
      {
        return mapping::offset( ext.next(), sum * ext.extent() + i, indices...);
      }

  public:

    constexpr index_type required_span_size() const noexcept
      { return m_extents.product(0,m_extents.rank()); }

    template<class ... Indices >
    constexpr
    typename std::enable_if<sizeof...(Indices) == Extents::rank(),index_type>::type
    operator()( Indices ... indices ) const noexcept
      { return mapping::offset( m_extents, 0, indices... ); }

/*
    template<class Index0, class Index1, class Index2 >
    typename std::enable_if< std::is_integral<Index0>::value &&
                             std::is_integral<Index1>::value &&
                             std::is_integral<Index2>::value &&
                             3 == Extents::rank() , index_type >::type
    operator()( Index0 i0 , Index1 i1 , Index2 i2 ) const noexcept
      { return ( ( ( i0 ) * m_extents.template extent<1>() + i1 )
                          * m_extents.template extent<2>() + i2 ); }
*/

    static constexpr bool is_always_unique     = true ;
    static constexpr bool is_always_contiguous = true ;
    static constexpr bool is_always_strided    = true ;

    constexpr bool is_unique()     const noexcept { return true ; }
    constexpr bool is_contiguous() const noexcept { return true ; }
    constexpr bool is_strided()    const noexcept { return true ; }

    constexpr index_type stride(size_t r) const noexcept
      { return m_extents.product(r+1,m_extents.rank()); }

  }; // struct mapping

}; // struct layout_right

}}} // std::experimental::fundamentals_v3

//----------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

struct layout_left {

  template<class Extents>
  class mapping {
  private:

    Extents m_extents ;

  public:

    using index_type = ptrdiff_t ;

    HOST_DEVICE
    constexpr mapping() noexcept = default ;

    HOST_DEVICE
    constexpr mapping( mapping && ) noexcept = default ;

    HOST_DEVICE
    constexpr mapping( const mapping & ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( mapping && ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( const mapping & ) noexcept = default ;

    HOST_DEVICE
    constexpr mapping( const Extents & ext ) noexcept
      : m_extents( ext ) {}

    constexpr const Extents & extents() const noexcept { return m_extents ; }

    template<class... Indices>
    constexpr mapping( Indices... DynamicExtents ) noexcept
      : m_extents( DynamicExtents... ) {}

  private:

    // ( i0 + N0 * ( i1 + N1 * ( i2 + N2 * ( ... ) ) ) )

    template<class Ext >
    HOST_DEVICE
    static constexpr index_type
    offset( const Ext & ) noexcept
      { return 0 ; }

    template<class Ext , class ... IndexType >
    HOST_DEVICE
    static constexpr index_type
    offset( const Ext & ext, index_type i, IndexType... indices ) noexcept
      { return i + ext.extent() * mapping::offset( ext.next(), indices... ); }

  public:

    constexpr index_type required_span_size() const noexcept
      { return m_extents.product(0,m_extents.rank()); }

    template<class ... Indices >
    constexpr
    typename std::enable_if<sizeof...(Indices) == Extents::rank(),index_type>::type
    operator()( Indices ... indices ) const noexcept
      { return mapping::offset( m_extents, indices... ); }

/*
    template<class Index0, class Index1, class Index2 >
    typename std::enable_if< std::is_integral<Index0>::value &&
                             std::is_integral<Index1>::value &&
                             std::is_integral<Index2>::value &&
                             3 == Extents::rank() , index_type >::type
    operator()( Index0 i0 , Index1 i1 , Index2 i2 ) const noexcept
      { return i0 + m_extents.template extent<0>() * (
               i1 + m_extents.template extent<1>() * ( i2 ) ); }
*/

    static constexpr bool is_always_unique     = true ;
    static constexpr bool is_always_contiguous = true ;
    static constexpr bool is_always_strided    = true ;

    constexpr bool is_unique()     const noexcept { return true ; }
    constexpr bool is_contiguous() const noexcept { return true ; }
    constexpr bool is_strided()    const noexcept { return true ; }

    constexpr index_type stride(size_t r) const noexcept
      { return m_extents.product(0,r); }

  }; // struct mapping

}; // struct layout_left

}}} // std::experimental::fundamentals_v3

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

struct layout_stride {

  template<class Extents>
  class mapping {
  private:

    using stride_t = std::array<ptrdiff_t,Extents::rank()> ;

    Extents   m_extents ;
    stride_t  m_stride ;
    int       m_contig ;
    int       m_unique ;

  public:

    using index_type = ptrdiff_t ;

    HOST_DEVICE
    constexpr mapping() noexcept = default ;

    HOST_DEVICE
    constexpr mapping( mapping && ) noexcept = default ;

    HOST_DEVICE
    constexpr mapping( const mapping & ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( mapping && ) noexcept = default ;

    HOST_DEVICE
    mapping & operator = ( const mapping & ) noexcept = default ;

    HOST_DEVICE
    mapping( const Extents & ext, const stride_t & str ) noexcept
      : m_extents(ext), m_stride(str), m_contig(1), m_unique(1)
      {
        int p[ Extents::rank() ? Extents::rank() : 1 ];

        // Fill permutation such that
        //   m_stride[ p[i] ] <= m_stride[ p[i+1] ]
        //
        for ( size_t i = 0 ; i < Extents::rank() ; ++i ) {

          int j = i ;

          while ( j && m_stride[i] < m_stride[ p[j-1] ] )
           { p[j] = p[j-1] ; --j ; }

          p[j] = i ;
        }

        for ( size_t i = 1 ; i < Extents::rank() ; ++i ) {
          const int j = p[i-1];
          const int k = p[i];
          const index_type prev = m_stride[j] * m_extents.extent(j);
          if ( m_stride[k] != prev ) { m_contig = 0 ; }
          if ( m_stride[k] <  prev ) { m_unique = 0 ; }
        }
      }

    constexpr const Extents & extents() const noexcept { return m_extents ; }

  private:

    // i0 * N0 + i1 * N1 + i2 * N2 + ...

    template<size_t>
    HOST_DEVICE
    constexpr index_type
    offset() const noexcept
      { return 0 ; }

    template<size_t K, class... IndexType >
    HOST_DEVICE
    constexpr index_type
    offset( index_type i, IndexType... indices ) const noexcept
      { return i * m_stride[K] + mapping::template offset<K+1>(indices...); }

  public:

    HOST_DEVICE
    index_type required_span_size() const noexcept
      {
        index_type max = 0 ;
        for ( size_t i = 0 ; i < Extents::rank() ; ++i )
          max += m_stride[i] * ( m_extents.extent(i) - 1 );
        return max ;
      }

    template<class ... Indices >
    constexpr
    typename std::enable_if<sizeof...(Indices) == Extents::rank(),index_type>::type
    operator()( Indices ... indices ) const noexcept
      { return mapping::offset( indices... ); }

/*
    template<class Index0, class Index1, class Index2 >
    typename std::enable_if< std::is_integral<Index0>::value &&
                             std::is_integral<Index1>::value &&
                             std::is_integral<Index2>::value &&
                             3 == Extents::rank() , index_type >::type
    operator()( Index0 i0 , Index1 i1 , Index2 i2 ) const noexcept
      { return i0 * m_stride_t[0] +
               i1 * m_stride_t[1] +
               i2 * m_stride_t[2] ; }
*/

    static constexpr bool is_always_unique     = false ;
    static constexpr bool is_always_contiguous = false ;
    static constexpr bool is_always_strided    = true ;

    constexpr bool is_unique()     const noexcept { return m_unique ; }
    constexpr bool is_contiguous() const noexcept { return m_contig ; }
    constexpr bool is_strided()    const noexcept { return true ; }

    constexpr index_type stride(size_t r) const noexcept
      { return m_stride[r]; }

  }; // struct mapping

}; // struct layout_stride

}}} // std::experimental::fundamentals_v3

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

template<class ElementType>
struct accessor_basic {
  using element_type  = ElementType;
  using pointer       = ElementType*;
  using handle_type   = ElementType*;
  using offset_policy = accessor_basic;
  using reference     = ElementType&;

  static typename offset_policy::handle_type
    offset( handle_type h , size_t i ) noexcept
      { return h+i; }

  static reference deref( handle_type h , size_t i ) noexcept
    { return h[i]; }

  static pointer decay( handle_type h ) noexcept
    { return h; }
};

}}} // std::experimental::fundamentals_v3

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

template<class T, size_t N>
class aligned_accessor {
public:
  static_assert( ( 0 == ( N & ( N - 1 ))), "" );
  static_assert( ( 0 == ( N % sizeof(T))), "" );

  enum : size_t { align = N };

  using element_type = T;
  using reference    = T&;
  using pointer      = T*;
  using offset       = T*;

  constexpr aligned_accessor() noexcept {};
  constexpr aligned_accessor( const aligned_accessor & ) noexcept = default ;

  aligned_accessor( aligned_accessor && ) = delete ;
  aligned_accessor operator = ( aligned_accessor && ) = delete ;
  aligned_accessor operator = ( const aligned_accessor & ) = delete ;

  explicit aligned_accessor( T * other ) noexcept
    : ptr(other)
    {
      // Verify pointer alignment:
      assert( 0 == reinterpret_cast<uintptr_t>(ptr) % N );
    }

  // constexpr operator [[aligned(N)]] T* () const noexcept { return ptr };

  constexpr operator T*() const noexcept
    { return ptr; }

  constexpr reference operator[]( size_t i ) const noexcept
    { return ptr[i]; }

  // Offsetting looses the alignment attribute
  constexpr offset operator+( size_t i ) const noexcept
    { return ptr+i; }

private:

  // [[aligned(N)]] T * const ptr = 0 ;
  T * const ptr = 0 ;
};

template<class ElementType, size_t N>
struct aligned_access_policy {
  using element_type  = ElementType;
  using pointer       = ElementType*;
  using handle_type   = aligned_accessor<ElementType,N>;
  using reference     = typename handle_type::reference;
  using offset_policy = accessor_basic<ElementType>;

  static typename offset_policy::handle_type
    offset( const handle_type & h , size_t i ) noexcept
      { return h+i; }

  static reference deref( const handle_type & h , size_t i ) noexcept
    { return h[i]; }

  static pointer decay( const handle_type & h ) noexcept
    { return (pointer)h; }
};

}}} // std::experimental::fundamentals_v3

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

namespace std {
namespace experimental {
namespace fundamentals_v3 {

template<class T, ptrdiff_t N = dynamic_extent >
using span = basic_mdspan<T,extents<N>,layout_none,accessor_basic<T> >;

template<class ElementType, class Extents, class LayoutPolicy, class AccessorPolicy>
class basic_mdspan {
public:

  // Domain and codomain types

  using extents_type     = Extents ;
  using layout_type      = LayoutPolicy ;
  using accessor_type    = AccessorPolicy ;
  using mapping_type     = typename layout_type::template mapping<extents_type> ;
  using element_type     = typename accessor_type::element_type ;
  using value_type       = typename std::remove_cv<element_type>::type ;
  using index_type       = ptrdiff_t ;
  using difference_type  = ptrdiff_t ;
  using handle_type      = typename accessor_type::handle_type;
  using pointer          = typename accessor_type::pointer;
  using reference        = typename accessor_type::reference;

  // [mdspan.basic.cons]

  HOST_DEVICE
  constexpr basic_mdspan() noexcept : m_acc(0), m_map() {}

  HOST_DEVICE
  constexpr basic_mdspan(basic_mdspan&& other) noexcept = default;

  HOST_DEVICE
  constexpr basic_mdspan(const basic_mdspan & other) noexcept = default;

  HOST_DEVICE
  basic_mdspan& operator=(const basic_mdspan & other) noexcept = default;

  HOST_DEVICE
  basic_mdspan& operator=(basic_mdspan&& other) noexcept = default;

  template<class OtherElementType,
           class OtherExtents,
           class OtherLayoutPolicy,
           class OtherAccessor>
  constexpr basic_mdspan(
    const basic_mdspan<OtherElementType,
                       OtherExtents,
                       OtherLayoutPolicy,
                       OtherAccessor> & rhs ) noexcept
    : m_acc( rhs.m_acc )
    , m_map( rhs.m_map )
    {}

  template<class OtherElementType,
           class OtherExtents,
           class OtherLayoutPolicy,
           class OtherAccessor>
  basic_mdspan & operator = (
    const basic_mdspan<OtherElementType,
                       OtherExtents,
                       OtherLayoutPolicy,
                       OtherAccessor> & rhs ) noexcept
    { m_acc = rhs.m_acc ; m_map = rhs.m_map ; return *this ; }

  template<class... IndexType >
  explicit constexpr basic_mdspan
    ( handle_type ptr , IndexType ... DynamicExtents ) noexcept
    : m_acc(ptr), m_map( DynamicExtents... ) {}

  constexpr basic_mdspan( handle_type ptr , const mapping_type & m ) noexcept
    : m_acc(ptr), m_map( m ) {}

  // [mdspan.basic.mapping]

  // Enforce rank() <= sizeof...(IndexType)
  template<class... IndexType >
  HOST_DEVICE
  constexpr
  typename std::enable_if<sizeof...(IndexType)==extents_type::rank(),reference>::type
  operator()( IndexType... indices) const noexcept
    { return m_acc[ m_map( indices... ) ]; }

  // Enforce rank() == 1
  template<class IndexType >
  HOST_DEVICE
  constexpr
  typename std::enable_if<std::is_integral<IndexType>::value && 1==extents_type::rank(),reference>::type
  operator[]( IndexType i ) const noexcept
    { return m_acc[ m_map(i) ]; }

  // [mdspan.basic.domobs]

  static constexpr int rank() noexcept
    { return extents_type::rank(); }

  static constexpr int rank_dynamic() noexcept
    { return extents_type::rank_dynamic(); }

  constexpr index_type static_extent( size_t k ) const noexcept
    { return m_map.extents().static_extent( k ); }

  constexpr index_type extent( int k ) const noexcept
    { return m_map.extents().extent( k ); }

  constexpr const extents_type & extents() const noexcept
    { return m_map.extents(); }

  // [mdspan.basic.codomain]

  // ------------------------------

  constexpr fundamentals_v3::span<element_type> span() const noexcept
	  { return fundamentals_v3::span<element_type>((pointer)m_acc,m_map.extents().extent(0)); }

  // ------------------------------

  // [mdspan.basic.obs]

  static constexpr bool is_always_unique = mapping_type::is_always_unique ;
  static constexpr bool is_always_regular = mapping_type::is_always_regular ;
  static constexpr bool is_always_contiguous = mapping_type::is_always_contiguous ;

  HOST_DEVICE
  constexpr bool is_unique() const noexcept  { return m_map.is_unique(); }
  HOST_DEVICE
  constexpr bool is_regular() const noexcept { return m_map.is_regular(); }
  HOST_DEVICE
  constexpr bool is_contiguous() const noexcept {return m_map.is_contiguous();}

  HOST_DEVICE
  constexpr index_type stride( size_t r ) const noexcept
    { return m_map.stride(r); }

  template<class... IndexType>
  HOST_DEVICE
  static constexpr
  typename std::enable_if<sizeof...(IndexType)==extents_type::rank_dynamic(),index_type>::type
  required_span_size(IndexType... DynamicExtents) noexcept
    { return mapping_type(DynamicExtents...).required_span_size(); }

  HOST_DEVICE
  constexpr const mapping_type & mapping() const noexcept { return m_map ; }

private:

  handle_type  m_acc ;
  mapping_type m_map ;
};


template<class T, ptrdiff_t... Indices>
using mdspan = basic_mdspan<T,extents<Indices...>,layout_right,accessor_basic<T> > ;

}}} // std::experimental::fundamentals_v3

//--------------------------------------------------------------------------

#if 0

namespace std {
namespace experimental {
namespace fundamentals_v3 {

template< class MDSPAN , typename ... SliceSpecs >
mdspan< typename MDSPAN::element_type
      , typename detail::sub_extents_deduction
          < typename MDSPAN::properties::extents
          , SliceSpecs...
          >::type
      , layout_stride
      >
subspan( MDSPAN const & a , SliceSpecs const & ... slice_specs )
{
  typedef typename MDSPAN::properties::extents
    extents_input ;

  typedef detail::sub_extents_deduction< extents_input , SliceSpecs...  >
    deduction ;

  typedef typename deduction::type
    extents_output ;

  typedef
    mdspan< typename MDSPAN::element_type
          , extents_output
          , layout_stride
          > return_type ;

  constexpr int output_rank = extents_output::rank();

  ptrdiff_t offset = a.offset( detail::slices_begin( slice_specs )... );

  ptrdiff_t dyn[ output_rank ? output_rank : 1 ];
  ptrdiff_t str[ output_rank ? output_rank : 1 ];

  deduction::get( dyn , str , a , slice_specs... );

  typedef typename
    detail::mdspan_mapping< extents_output , layout_stride >::type
      mapping ;
  
  return return_type( a.data() + offset , mapping( dyn , str ) );
}

}}} // std::experimental::fundamentals_v3

#endif /* #if 0 */

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

#endif // #ifndef STD_EXPERIMENTAL_MDSPAN_HEADER
